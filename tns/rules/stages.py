"""Машина стадий S0–S5 (RULE-3).

Для каждого ЛС на каждый месяц определяет стадию взыскания на 01.M, до мер месяца M.

Как считается стадия месяца M:
1. Стадия по фактам: самая тяжёлая мера текущего цикла, применённая до 01.M
   (какой стадии соответствует мера — configs/settings.yaml, stages.measure_stage).
2. База: бо́льшая из стадии по фактам и стадии прошлого месяца. Откатов нет.
3. Погашение: если сработал переход в S0 (T-*→0, «долг < P-01»), ЛС уходит в S0,
   цикл закрывается, даты и факты цикла сбрасываются.
4. Иначе не больше одного шага вперёд: первый сработавший переход с листа «Переходы»
   (T-0→1, T-1→2, T-2→3, …). Условия переходов берутся из таблицы правил и вычисляются
   модулем tns.rules.conditions.

Цикл — от появления долга (стадия выходит из S0) до погашения (возврат в S0).
Меры прошлых циклов на стадию не влияют.

Историю за все месяцы считаем один раз и кладём в кэш (output/cache/stages_history.parquet).
Пока панель, журнал мер, таблица правил и настройки не менялись, следующий запуск берёт готовое.
Для людей пишется output/<ГГГГ-ММ>/stages.xlsx: «Сводка», «Текущая стадия», «Переходы».

Формат таблицы stages — docs/interfaces.md, раздел 4.

Запуск отдельно от пайплайна:
    python -m tns.rules.stages --month 2026-02 [--data ПАПКА] [--rules КНИГА.xlsx] [--out ПАПКА] [--no-cache]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tns.rules.conditions import Condition, Env, compile_condition, params_of
from tns.rules.loader import RuleBook, Transition, load_rulebook

logger = logging.getLogger(__name__)

# Меняйте при любом изменении алгоритма: старый кэш истории станет недействительным.
STAGES_VERSION = "1"

HISTORY_FILE = "stages_history"
STAGE_SOURCES = ["start", "fact", "rule", "carry"]
EXCEL_MAX_ROWS = 1_048_575  # предел листа Excel без строки заголовков
FACT_LABEL = "по факту"  # код изменения стадии по применённой мере (не по переходу из таблицы правил)

# Какие даты цикла обновляет мера и как: первая дата в цикле или последняя.
DATE_FIELDS: dict[str, tuple[str, str]] = {
    "restriction_notice": ("notice_date", "first"),
    "restriction": ("restriction_date", "last"),
    "court_application": ("court_application_date", "last"),
    "court_result": ("court_result_date", "last"),
    "claim": ("last_claim_date", "last"),
}
# Результат меры (P-03) для условий result(…): предмет → мера и поле даты.
RESULT_SUBJECTS = {"notice": "restriction_notice", "restriction": "restriction"}
# Последняя претензия нужна для «не чаще P-09» и при погашении не сбрасывается.
KEEP_AFTER_CLOSE = {"last_claim_date"}
CYCLE_DATES = [f for f, _ in DATE_FIELDS.values()]
DATE_LABELS = {
    "notice_date": "уведомление",
    "restriction_date": "ограничение",
    "court_application_date": "заявление в суд",
    "court_result_date": "приказ / ИЛ",
    "last_claim_date": "претензия",
}
PANEL_FIELDS = ["dz", "overdue_m", "norm", "months_no_payment", "dz_overdue"]

STAGES_COLUMNS = [
    "ls", "month", "stage", "stage_prev", "stage_fact", "stage_source", "months_in_stage", "transition",
    "exception_codes", "cycle_id", "cycle_start", *CYCLE_DATES, "reason", "rules_version",
]


class StagesError(ValueError):
    """Ошибка в настройках машины стадий или в таблице правил. Сообщение — по-русски."""


# ---------------------------------------------------------------------------
# Модель: стадии и переходы из таблицы правил
# ---------------------------------------------------------------------------


@dataclass
class _Step:
    transition: Transition
    condition: Condition
    from_levels: list[int]
    to_level: int


@dataclass
class StageModel:
    levels: list[str]  # S0 … S5
    names: dict[str, str]
    closure: _Step
    forward: list[_Step]
    codes: list[str]  # коды переходов, индекс = номер в массиве transition
    measure_level: dict[str, int]
    court_rules: list = field(default_factory=list)
    field_defaults: dict = field(default_factory=dict)
    p01: float = 0.0
    rules_version: str = ""


def build_model(book: RuleBook, settings: dict) -> StageModel:
    cfg = settings.get("stages") or {}
    levels = sorted((c for c in book.stages if re.fullmatch(r"S\d", c)), key=lambda c: int(c[1:]))
    if not levels or levels[0] != "S0":
        raise StagesError("на листе «Стадии» нет стадии S0")
    level_of = {s: i for i, s in enumerate(levels)}
    params = params_of(book)

    closure, forward = None, []
    for t in book.transitions:
        if t.returns_to_previous or t.to_stage not in level_of:  # SX и возврат из SX — исключения
            if cfg.get("exceptions"):
                raise StagesError("исключения (stages.exceptions: true) пока не поддерживаются машиной стадий")
            continue
        if not t.condition:
            raise StagesError(f"у перехода {t.code} нет формального условия на листе «Переходы»")
        to = level_of[t.to_stage]
        froms = [level_of[s] for s in t.from_stages if s in level_of]
        step = _Step(t, compile_condition(t.condition, params, f"переход {t.code}"), froms, to)
        if to == 0:
            closure = step
        elif froms and all(f < to for f in froms):
            forward.append(step)
        elif cfg.get("rollbacks"):
            raise StagesError("откаты (stages.rollbacks: true) пока не поддерживаются машиной стадий")
    if closure is None:
        raise StagesError("на листе «Переходы» нет перехода в S0 (погашение долга)")

    measure_level = {}
    for measure, stage in (cfg.get("measure_stage") or {}).items():
        if stage not in level_of:
            raise StagesError(f"configs/settings.yaml, stages.measure_stage: у меры «{measure}» неизвестная стадия «{stage}»")
        measure_level[measure] = level_of[stage]
    if not measure_level:
        raise StagesError("configs/settings.yaml: не заполнен раздел stages.measure_stage")

    if params.get("P-01") is None:
        raise StagesError("на листе «Параметры» нет числового P-01 (минимальная ДЗ)")
    version = book.version
    return StageModel(
        levels=levels,
        names={c: s.name for c, s in book.stages.items()},
        closure=closure,
        forward=forward,
        codes=[closure.transition.code] + [s.transition.code for s in forward],
        measure_level=measure_level,
        court_rules=book.rules_for_measure("court_application"),
        field_defaults=dict(cfg.get("field_defaults") or {}),
        p01=float(params["P-01"]),
        rules_version=f"v{version.label or '?'} ({version.sha256[:12]})",
    )


def _court_threshold_ok(model: StageModel, dz: np.ndarray, overdue: np.ndarray) -> np.ndarray:
    """Выполнены пороги хотя бы одного активного правила подачи в суд (R-CRT-01…03)."""
    ok = np.zeros(len(dz), dtype=bool)
    with np.errstate(invalid="ignore"):
        for r in model.court_rules:
            m = np.ones(len(dz), dtype=bool)
            if r.overdue_from is not None:
                m &= overdue >= r.overdue_from
            if r.overdue_to is not None:
                m &= overdue <= r.overdue_to
            if r.dz_from is not None:
                m &= dz >= r.dz_from
            if r.dz_to is not None:
                m &= dz <= r.dz_to
            ok |= m
    return ok


# ---------------------------------------------------------------------------
# Проход по месяцам
# ---------------------------------------------------------------------------


def _grid(panel: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Панель → месяцы, ЛС и массивы [месяц, ЛС] для нужных колонок."""
    months = np.sort(panel["month"].unique())
    ls = np.sort(panel["ls"].unique())
    if len(panel) != len(months) * len(ls):
        idx = pd.MultiIndex.from_product([months, ls], names=["month", "ls"])
        panel = panel.set_index(["month", "ls"]).reindex(idx).reset_index()
    else:
        panel = panel.sort_values(["month", "ls"], ignore_index=True)
    shape = (len(months), len(ls))
    arrays = {c: panel[c].to_numpy(dtype=float, na_value=np.nan).reshape(shape) for c in PANEL_FIELDS}
    arrays["dz"] = np.nan_to_num(arrays["dz"], nan=0.0)
    arrays["pay30_m"] = panel["pay30_m"].astype("boolean").fillna(False).to_numpy(dtype=bool).reshape(shape)
    return months, ls, arrays


def _events_by_month(events: pd.DataFrame, months: np.ndarray, ls: np.ndarray) -> dict[int, pd.DataFrame]:
    """События, сгруппированные по номеру месяца; для ЛС — позиция в массиве ls."""
    e = events[events["ls"].isin(ls) & events["month"].isin(months)].copy()
    e["pos"] = np.searchsorted(ls, e["ls"].to_numpy())
    e["t"] = np.searchsorted(months, e["month"].to_numpy())
    e["res"] = e["result_30pct"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    return {int(t): g for t, g in e.groupby("t")}


def build_history(panel: pd.DataFrame, events: pd.DataFrame, book: RuleBook, settings: dict) -> pd.DataFrame:
    """Стадии всех ЛС за все месяцы панели. Каждый месяц M использует только данные до 01.M."""
    model = build_model(book, settings)
    months, ls, A = _grid(panel)
    T, n = len(months), len(ls)
    ev = _events_by_month(events, months, ls)
    nat = np.datetime64("NaT", "ns")

    stage = np.full(n, -1, dtype=np.int8)  # −1: месяца раньше не было
    months_in = np.zeros(n, dtype=np.int64)
    cycle_id = np.zeros(n, dtype=np.int64)
    cycle_start = np.full(n, nat)
    fact = np.zeros(n, dtype=np.int8)
    dates = {f: np.full(n, nat) for f in CYCLE_DATES}
    results = {s: np.zeros(n, dtype=bool) for s in RESULT_SUBJECTS}
    defaults = {k: np.full(n, bool(v)) if isinstance(v, bool) else np.full(n, v) for k, v in model.field_defaults.items()}

    out: dict[str, list] = {k: [] for k in (
        "stage", "stage_prev", "stage_fact", "source", "months", "transition", "cycle_id", "cycle_start", *CYCLE_DATES)}

    for t in range(T):
        today = np.datetime64(months[t], "ns")

        # 1. Факты прошлого месяца: меры месяца M−1 известны на 01.M.
        g = ev.get(t - 1)
        if g is not None:
            for measure, sub in g.groupby("measure", observed=True):
                pos = sub["pos"].to_numpy()
                level = model.measure_level.get(str(measure))
                if level is not None:
                    fact[pos] = np.maximum(fact[pos], level)
                if str(measure) in DATE_FIELDS:
                    name, how = DATE_FIELDS[str(measure)]
                    d = sub["event_date"].to_numpy(dtype="datetime64[ns]")
                    mask = np.isnat(dates[name][pos]) if how == "first" else np.ones(len(pos), dtype=bool)
                    dates[name][pos[mask]] = d[mask]
                    for subject, m_key in RESULT_SUBJECTS.items():
                        if m_key == str(measure):
                            results[subject][pos[mask]] = sub["res"].to_numpy()[mask]

        # 2. База: стадия прошлого месяца или по фактам — что выше.
        prev = stage.copy()
        prev_level = np.maximum(prev, 0)
        base = np.maximum(prev_level, fact)
        months_before = np.where(prev == base, months_in, 0)

        dz, overdue = A["dz"][t], A["overdue_m"][t]
        env = Env(
            fields={
                **{c: A[c][t] for c in PANEL_FIELDS}, "dz": dz,
                "months_in_stage": months_before.astype(float),
                "court_threshold_ok": _court_threshold_ok(model, dz, overdue),
                **dates, **defaults,
            },
            results={"info": A["pay30_m"][t - 1] if t > 0 else np.zeros(n, dtype=bool), **results},
            today=today,
            size=n,
        )

        # 3. Погашение, затем не больше одного шага вперёд.
        closing = model.closure.condition.evaluate(env)
        new = base.copy()
        trans = np.full(n, -1, dtype=np.int16)
        applied = closing.copy()
        for i, step in enumerate(model.forward, start=1):
            cand = ~applied & np.isin(base, step.from_levels)
            if not cand.any():
                continue
            ok = cand & step.condition.evaluate(env)
            new[ok] = step.to_level
            trans[ok] = i
            applied |= ok
        new[closing] = 0
        trans[closing & (prev > 0)] = 0

        if t == 0:
            source = np.full(n, 0, dtype=np.int8)  # start: до первого месяца данных истории нет
            trans[:] = -1
        else:
            source = np.where(trans >= 0, 2, np.where(new > prev_level, 1, 3)).astype(np.int8)

        opened = (new > 0) & (prev <= 0)
        cycle_id[opened] += 1
        cycle_start[opened] = today
        months_in = np.where(new == prev, months_in + 1, 1)

        # 4. Закрытый цикл: факты и даты цикла сбрасываются.
        fact_now = fact.copy()
        if closing.any():
            fact[closing] = 0
            for name in CYCLE_DATES:
                if name not in KEEP_AFTER_CLOSE:
                    dates[name][closing] = nat
            for subject in results:
                results[subject][closing] = False
        stage = new

        out["stage"].append(new.copy())
        out["stage_prev"].append(prev)
        out["stage_fact"].append(fact_now)
        out["source"].append(source)
        out["months"].append(months_in.copy())
        out["transition"].append(trans)
        out["cycle_id"].append(cycle_id.copy())
        out["cycle_start"].append(np.where(new > 0, cycle_start, nat))
        for name in CYCLE_DATES:
            out[name].append(dates[name].copy())

    return _to_frame(out, months, ls, model)


def _to_frame(out: dict, months: np.ndarray, ls: np.ndarray, model: StageModel) -> pd.DataFrame:
    stage_codes = model.levels + ["SX"]

    def cat(values, categories):
        return pd.Categorical.from_codes(values.astype(np.int16), categories=categories)

    prev = np.concatenate(out["stage_prev"])
    trans = np.concatenate(out["transition"])
    code_arr = np.array(model.codes + [None], dtype=object)
    df = pd.DataFrame({
        "ls": np.tile(ls, len(months)).astype("int64"),
        "month": np.repeat(months, len(ls)).astype("datetime64[ns]"),
        "stage": cat(np.concatenate(out["stage"]), stage_codes),
        "stage_prev": cat(prev, stage_codes),  # −1 → пусто
        "stage_fact": cat(np.concatenate(out["stage_fact"]), stage_codes),
        "stage_source": cat(np.concatenate(out["source"]), STAGE_SOURCES),
        "months_in_stage": np.concatenate(out["months"]).astype("int64"),
        "transition": pd.array(code_arr[trans], dtype="string"),
        "exception_codes": pd.array([pd.NA] * len(trans), dtype="string"),
        "cycle_id": np.concatenate(out["cycle_id"]).astype("int64"),
        "cycle_start": np.concatenate(out["cycle_start"]),
        **{name: np.concatenate(out[name]) for name in CYCLE_DATES},
        "reason": pd.array([pd.NA] * len(trans), dtype="string"),
        "rules_version": pd.array([model.rules_version] * len(trans), dtype="string"),
    })
    return df[STAGES_COLUMNS]


# ---------------------------------------------------------------------------
# Объяснения
# ---------------------------------------------------------------------------


def _rub(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ") + " ₽" if pd.notna(x) else "нет данных"


def _months(x) -> str:
    return f"{int(x)} мес." if pd.notna(x) else "нет данных"


def _date(x) -> str:
    return f"{pd.Timestamp(x):%d.%m.%Y}"


def explain(rows: pd.DataFrame, panel: pd.DataFrame, book: RuleBook, settings: dict) -> pd.Series:
    """Объяснение стадии по-русски для каждой строки rows (таблица stages)."""
    model = build_model(book, settings)
    by_code = {t.code: t for t in book.transitions}
    level_dates = {}  # стадия → поле даты меры этой стадии (для «по факту»)
    for measure, level in model.measure_level.items():
        if measure in DATE_FIELDS and DATE_FIELDS[measure][0] != "last_claim_date":
            level_dates.setdefault(model.levels[level], DATE_FIELDS[measure][0])

    p = panel.set_index(["ls", "month"])[["dz", "overdue_m"]]
    joined = rows.join(p, on=["ls", "month"])
    texts = []
    for r in joined.itertuples(index=False):
        money = f"долг {_rub(r.dz)}, просрочка {_months(r.overdue_m)}"
        cycle_dates = "; ".join(
            f"{DATE_LABELS[f]} {_date(getattr(r, f))}" for f in CYCLE_DATES
            if f != "last_claim_date" and pd.notna(getattr(r, f)))
        tail = f"; {cycle_dates}" if cycle_dates else ""
        stage = str(r.stage)
        name = model.names.get(stage, stage)
        if pd.notna(r.transition):
            t = by_code.get(r.transition)
            human = t.condition_text if t and t.condition_text else ""
            texts.append(f"{r.transition}: {human} — {money}{tail}")
        elif r.stage_source == "fact":
            f = level_dates.get(stage)
            what = f"{DATE_LABELS[f]} {_date(getattr(r, f))}" if f and pd.notna(getattr(r, f)) else "применялись меры информирования"
            texts.append(f"{stage} «{name}» по факту: {what} — {money}")
        elif stage == "S0":
            low = pd.notna(r.dz) and r.dz < model.p01
            texts.append(f"S0 «{name}»: долг {_rub(r.dz)}" + (f" < {_rub(model.p01)}" if low else ", просрочки нет"))
        else:
            start = "начало данных" if r.stage_source == "start" else f"{int(r.months_in_stage)} мес. в стадии"
            texts.append(f"{stage} «{name}», {start} — {money}{tail}")
    return pd.Series(pd.array(texts, dtype="string"), index=rows.index)


# ---------------------------------------------------------------------------
# Кэш истории
# ---------------------------------------------------------------------------


def _frame_hash(df: pd.DataFrame, columns: list[str]) -> str:
    sub = df[columns].sort_values(columns[:2], ignore_index=True)
    return hashlib.sha256(pd.util.hash_pandas_object(sub, index=False).to_numpy().tobytes()).hexdigest()


def history_key(panel: pd.DataFrame, events: pd.DataFrame, book: RuleBook, settings: dict) -> dict:
    return {
        "version": STAGES_VERSION,
        "rules_sha256": book.version.sha256,
        "settings": json.dumps(settings.get("stages") or {}, sort_keys=True, ensure_ascii=False),
        "panel": _frame_hash(panel, ["ls", "month", *PANEL_FIELDS, "pay30_m"]),
        "events": _frame_hash(events, ["ls", "month", "measure", "event_date", "result_30pct"]),
    }


def load_or_build_history(panel: pd.DataFrame, events: pd.DataFrame, book: RuleBook, settings: dict,
                          cache_dir: Path | None) -> pd.DataFrame:
    """История стадий: из кэша, если входы не менялись, иначе пересчёт и запись в кэш."""
    key = history_key(panel, events, book, settings)
    if cache_dir is not None:
        table_path, meta_path = Path(cache_dir) / f"{HISTORY_FILE}.parquet", Path(cache_dir) / f"{HISTORY_FILE}.json"
        if table_path.exists() and meta_path.exists():
            try:
                if json.loads(meta_path.read_text(encoding="utf-8")) == key:
                    history = pd.read_parquet(table_path)
                    logger.info("Стадии: история взята из кэша «%s» (%s строк).", table_path, f"{len(history):,}".replace(",", " "))
                    return history
            except Exception as error:  # noqa: BLE001 — испорченный кэш просто пересчитываем
                logger.warning("Кэш стадий «%s» не прочитался (%s), считаю заново.", table_path, error)

    started = time.perf_counter()
    history = build_history(panel, events, book, settings)
    changed = history["stage_source"].isin(["rule", "fact"])
    history.loc[changed, "reason"] = explain(history.loc[changed], panel, book, settings)
    logger.info("Стадии: история посчитана за %.1f с (%s строк).", time.perf_counter() - started,
                f"{len(history):,}".replace(",", " "))

    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        clean = history.copy(deep=False)
        clean.attrs = {}
        clean.to_parquet(table_path, index=False)
        meta_path.write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    return history


# ---------------------------------------------------------------------------
# Excel для людей
# ---------------------------------------------------------------------------


def _write_sheet(wb, title: str, header: list[str], rows, widths: list[int]):
    from openpyxl.cell import WriteOnlyCell
    from openpyxl.styles import Font

    ws = wb.create_sheet(title)
    for i, w in enumerate(widths):
        ws.column_dimensions[chr(ord("A") + i)].width = w
    ws.freeze_panes = "A2"
    head = []
    for h in header:
        c = WriteOnlyCell(ws, value=h)
        c.font = Font(bold=True)
        head.append(c)
    ws.append(head)
    for row in rows:
        ws.append(row)


def write_excel(stages: pd.DataFrame, month: pd.Timestamp, path: Path, book: RuleBook, settings: dict) -> Path:
    """stages.xlsx: «Сводка», «Текущая стадия» (месяц плана), «Переходы» (вся история до месяца плана)."""
    from openpyxl import Workbook

    model = build_model(book, settings)
    cur = stages[stages["month"] == month]
    # Изменения стадии: переход по правилу (код T-…) или по факту применённой меры.
    trans = stages[stages["stage_source"].isin(["rule", "fact"])].copy()
    trans["code"] = trans["transition"].fillna(FACT_LABEL)
    last = trans.sort_values("month").groupby("ls").tail(1).set_index("ls")[["code", "month"]]
    cur = cur.join(last.rename(columns={"code": "last_tr", "month": "last_tr_month"}), on="ls")

    wb = Workbook(write_only=True)

    counts = cur["stage"].value_counts().reindex(model.levels, fill_value=0)
    month_trans = cur["transition"].value_counts()
    summary = [[f"Месяц плана: {month:%Y-%m}", None, None], [f"Таблица правил: {model.rules_version}", None, None], [None, None, None],
               ["Стадия", "Название", "ЛС"]]
    summary += [[s, model.names.get(s, ""), int(counts[s])] for s in model.levels]
    summary += [[None, None, None], ["Переход в месяце плана", "", "ЛС"]]
    summary += [[code, "", int(v)] for code, v in month_trans.items()]
    pivot = stages.pivot_table(index="month", columns="stage", values="ls", aggfunc="count", observed=False)
    summary += [[None, None, None], ["Месяц", *model.levels]]
    summary += [[f"{m:%Y-%m}", *[int(pivot.loc[m, s]) if s in pivot.columns else 0 for s in model.levels]] for m in pivot.index]
    _write_sheet(wb, "Сводка", summary[0], summary[1:], [24, 30, 10, 10, 10, 10, 10])

    def cur_rows():
        for r in cur.itertuples(index=False):
            yield [int(r.ls), str(r.stage), model.names.get(str(r.stage), ""), int(r.months_in_stage), int(r.cycle_id),
                   None if pd.isna(r.cycle_start) else f"{r.cycle_start:%Y-%m}",
                   None if pd.isna(r.last_tr) else str(r.last_tr),
                   None if pd.isna(r.last_tr_month) else f"{r.last_tr_month:%Y-%m}",
                   None if pd.isna(r.reason) else str(r.reason)]

    _write_sheet(wb, "Текущая стадия",
                 ["ЛС", "Стадия", "Название", "Месяцев в стадии", "Цикл", "Начало цикла", "Последнее изменение",
                  "Месяц изменения", "Объяснение"], cur_rows(), [12, 8, 22, 10, 7, 12, 12, 12, 100])

    if len(trans) > EXCEL_MAX_ROWS:
        logger.warning("Переходов %d, в лист Excel помещаются последние %d.", len(trans), EXCEL_MAX_ROWS)
        trans = trans.sort_values("month").tail(EXCEL_MAX_ROWS)

    def trans_rows():
        for r in trans.sort_values(["ls", "month"]).itertuples(index=False):
            yield [int(r.ls), f"{r.month:%Y-%m}", None if pd.isna(r.stage_prev) else str(r.stage_prev), str(r.stage),
                   str(r.code), None if pd.isna(r.reason) else str(r.reason)]

    _write_sheet(wb, "Переходы", ["ЛС", "Месяц", "Было", "Стало", "Код", "Объяснение"], trans_rows(),
                 [12, 9, 6, 6, 10, 110])
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


# ---------------------------------------------------------------------------
# Шаг пайплайна
# ---------------------------------------------------------------------------


def build_stages(ctx) -> pd.DataFrame:
    """Шаг «стадии» пайплайна (tns/pipeline.py). Возвращает таблицу stages за месяцы до месяца плана включительно."""
    panel, events = ctx.tables["panel"], ctx.tables["measure_events"]
    book = load_rulebook(ctx.rules_path)  # версия и предупреждения таблицы пишутся в журнал загрузчиком
    history = load_or_build_history(panel, events, book, ctx.settings, ctx.cache_dir)

    month = pd.Timestamp(ctx.month)
    available = history["month"].unique()
    if month not in set(pd.to_datetime(available)):
        raise StagesError(
            f"нет стадий на {month:%Y-%m}: панель покрывает {pd.Timestamp(available.min()):%Y-%m}…"
            f"{pd.Timestamp(available.max()):%Y-%m}. Укажите месяц из этого периода."
        )
    stages = history[history["month"] <= month].reset_index(drop=True)

    plan = (stages["month"] == month) & stages["reason"].isna()
    stages.loc[plan, "reason"] = explain(stages.loc[plan], panel, book, ctx.settings)

    cur = stages[stages["month"] == month]
    counts = cur["stage"].value_counts().reindex([c for c in cur["stage"].cat.categories if c != "SX"], fill_value=0)
    logger.info("Стадии на %s: %s.", f"{month:%Y-%m}", ", ".join(f"{s} — {n:,}".replace(",", " ") for s, n in counts.items()))
    moved = cur["transition"].value_counts()
    if len(moved):
        logger.info("Переходы на %s: %s.", f"{month:%Y-%m}", ", ".join(f"{c} — {n:,}".replace(",", " ") for c, n in moved.items()))

    if (ctx.settings.get("stages") or {}).get("excel", True):
        started = time.perf_counter()
        path = write_excel(stages, month, Path(ctx.out_dir) / "stages.xlsx", book, ctx.settings)
        logger.info("Стадии для просмотра: «%s» (%.0f с).", path, time.perf_counter() - started)
    return stages


# ---------------------------------------------------------------------------
# Командная строка
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    from tns.data import panel as panel_module
    from tns.data.loaders import REPO_ROOT, default_cache_dir, load_settings
    from tns.logs import report_error, setup_logging
    from tns.pipeline import RunContext

    parser = argparse.ArgumentParser(prog="python -m tns.rules.stages",
                                     description="Считает стадии взыскания S0–S5 по всем ЛС и пишет stages.parquet и stages.xlsx.")
    parser.add_argument("--month", required=True, help="месяц плана ГГГГ-ММ, например 2026-02")
    parser.add_argument("--data", help="папка с файлами 01–14 (по умолчанию из configs/settings.yaml)")
    parser.add_argument("--rules", help="таблица правил .xlsx (по умолчанию из configs/settings.yaml)")
    parser.add_argument("--out", help="папка результатов (по умолчанию output)")
    parser.add_argument("--config", help="файл настроек (по умолчанию configs/settings.yaml)")
    parser.add_argument("--no-cache", action="store_true", help="не брать кэш: перечитать файлы и пересчитать историю")
    args = parser.parse_args(argv)
    log_path = setup_logging("stages", argv=argv)
    try:
        settings = load_settings(args.config)
        month = panel_module.parse_plan_month(args.month)
        cache_dir = None if args.no_cache else default_cache_dir(settings)
        data_dir = Path(args.data) if args.data else REPO_ROOT / settings["paths"]["data_dir"]
        out_root = Path(args.out) if args.out else REPO_ROOT / settings["paths"]["output_dir"]
        tables, _ = panel_module.build_tables(data_dir, settings, cache_dir)
        ctx = RunContext(month=month, data_dir=data_dir, out_dir=out_root / f"{month:%Y-%m}",
                         rules_path=Path(args.rules) if args.rules else REPO_ROOT / settings["paths"]["rules"],
                         settings=settings, cache_dir=cache_dir, tables=tables)
        stages = build_stages(ctx)
        ctx.out_dir.mkdir(parents=True, exist_ok=True)
        panel_module.write_parquet(stages, ctx.out_dir / "stages.parquet")
        logger.info("Таблица стадий: «%s».", ctx.out_dir / "stages.parquet")
    except Exception as error:  # noqa: BLE001 — любую ошибку объясняем пользователю
        report_error(error, log_path, "Стадии не посчитаны: ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
