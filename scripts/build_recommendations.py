#!/usr/bin/env python3
"""
Пункты 2.3, 3 и 4 стратегии: Рекомендательная система мер воздействия на должников.

Принимает на вход:
  - output/kprototypes_features.parquet (или .csv)  — признаки ЛС
  - output/utility_matrix.parquet                   — uplift + expected_recovery_gain
  - output/kprototypes_labels.csv                   — сегменты
  - Материалы к кейсу/04..13 xlsx                  — история мер (для state machine)
  - configs/rules.yaml                              — Приложение №1
  - configs/limits.yaml                             — Приложение №2
  - configs/costs.yaml                              — cost_tier мер

Производит:
  output/recommendations_YYYY-MM.csv       — персональный план на месяц
  output/recommendations_summary.csv       — агрегаты по мерам и сегментам

Запуск:
  PYTHONPATH=/tmp/pylibs python3 scripts/build_recommendations.py \\
      --data-dir "Материалы к кейсу" \\
      --features output/kprototypes_features.parquet \\
      --utility  output/utility_matrix.parquet \\
      --labels   output/kprototypes_labels.csv \\
      --config   configs/ \\
      --month    2026-02 \\
      --out      output/recommendations_2026-02.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Справочники (соответствие внутренних ключей и файлов мер)
# ---------------------------------------------------------------------------

MEASURE_FILES = {
    "autodial":           "04 Автодозвон ХК.xlsx",
    "email":              "05 E-mail ХК.xlsx",
    "sms":                "06 СМС ХК.xlsx",
    "operator_call":      "07 Обзвон оператором ХК.xlsx",
    "claim":              "08 Претензия ХК.xlsx",
    "visit":              "09 Выезд к абоненту ХК.xlsx",
    "restriction_notice": "10 Уведомление о введении ограничения ХК.xlsx",
    "restriction":        "11 Ограничение ХК.xlsx",
    "court_application":  "12 Заявление о выдаче судебного приказа ХК.xlsx",
    "court_result":       "13 Получение судебного приказа или ИЛ ХК.xlsx",
}

MEASURE_RU = {
    "autodial":           "Автодозвон",
    "email":              "E-mail рассылка",
    "sms":                "СМС",
    "operator_call":      "Обзвон оператором",
    "claim":              "Претензия",
    "visit":              "Выезд к абоненту",
    "restriction_notice": "Уведомление об ограничении",
    "restriction":        "Ограничение подачи электроэнергии",
    "court_application":  "Заявление о выдаче судебного приказа",
    "court_result":       "Получение судебного приказа / ИЛ",
}

SEGMENT_NAMES = {
    0: "Хроническая просрочка (в суде/ограничении)",
    1: "Средняя задолженность (уведомление ограничения)",
    2: "Временные трудности (высокое ДЗ, хорошая история)",
    3: "Незначительная задолженность",
    4: "Без задолженности",
    5: "Мелкая задержка (только информирование)",
    -1: "Неопределённый сегмент",
}

# Информирующие меры (применяются постоянно совместно с другими)
BACKGROUND_MEASURES = {"email", "autodial"}
# Ведущие меры (одна на ЛС, по приоритету utility)
LEADING_MEASURES = {
    "sms", "operator_call", "claim", "visit",
    "restriction_notice", "restriction", "court_application",
}
# Все оперативные меры в order priority (от лёгких к тяжёлым)
ALL_MEASURES = list(MEASURE_FILES.keys())


# ---------------------------------------------------------------------------
# Загрузка конфигов
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    if yaml is None:
        raise ImportError("pip install pyyaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_configs(config_dir: Path) -> Tuple[dict, dict, dict]:
    """Возвращает (rules, limits, costs)."""
    rules = _load_yaml(config_dir / "rules.yaml").get("measures", {})
    limits_raw = _load_yaml(config_dir / "limits.yaml").get("limits", {})
    costs_cfg = _load_yaml(config_dir / "costs.yaml")
    cost_tiers = costs_cfg.get("cost_tiers", {})
    cost_rub = costs_cfg.get("cost_rub") or {}

    limits: Dict[str, Optional[int]] = {}
    for m, v in limits_raw.items():
        limits[m] = int(v) if v is not None else None

    return rules, limits, {"tiers": cost_tiers, "rub": cost_rub}


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_features(features_path: Path) -> pd.DataFrame:
    if features_path.suffix == ".parquet":
        df = pd.read_parquet(features_path)
    else:
        df = pd.read_csv(features_path, low_memory=False)
    df["ls"] = pd.to_numeric(df["ls"], errors="coerce").astype("Int64")
    return df


def load_utility(utility_path: Path) -> pd.DataFrame:
    if utility_path.suffix == ".parquet":
        df = pd.read_parquet(utility_path)
    else:
        df = pd.read_csv(utility_path, low_memory=False)
    df["ls"] = pd.to_numeric(df["ls"], errors="coerce").astype("Int64")
    return df


def load_labels(labels_path: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_path)
    df["ls"] = pd.to_numeric(df["ls"], errors="coerce").astype("Int64")
    return df[["ls", "segment"]]


def load_measure_history(data_dir: Path) -> pd.DataFrame:
    """
    Загружает историю всех мер воздействия в long-формат.
    Возвращает DataFrame с колонками: ls, measure, event_date.
    """
    records = []
    for measure_key, fname in MEASURE_FILES.items():
        path = data_dir / fname
        if not path.exists():
            continue
        try:
            df = pd.read_excel(path, header=1)
            df.columns = ["ls", "event_date"]
            df["ls"] = pd.to_numeric(df["ls"], errors="coerce").astype("Int64")
            df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
            df = df.dropna(subset=["ls", "event_date"])
            df["measure"] = measure_key
            records.append(df[["ls", "measure", "event_date"]])
        except Exception as e:
            log.warning("Ошибка чтения %s: %s", fname, e)
    if records:
        return pd.concat(records, ignore_index=True)
    return pd.DataFrame(columns=["ls", "measure", "event_date"])


# ---------------------------------------------------------------------------
# State machine: определение стадии для каждого ЛС
# ---------------------------------------------------------------------------

def compute_procedure_state(
    history: pd.DataFrame,
    payments: Optional[pd.DataFrame] = None,
    as_of: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    По истории мер определяет для каждого ЛС:
      stage            : достигнутая стадия (informing / restriction_notice / restriction / court / unknown)
      ever_restricted  : было ли ограничение подачи
      ever_in_court    : было ли судебное заявление
      had_informing    : было ли информирование (любой вид)
      can_do_restriction  : разрешено переходить к ограничению (информирование применялось)
      can_do_court        : разрешено переходить к суду (ограничение применялось или невозможно)
    """
    if as_of is None:
        as_of = pd.Timestamp.now()

    informing_measures = {"sms", "email", "autodial", "operator_call", "claim", "restriction_notice"}
    restriction_measures = {"restriction"}
    court_measures = {"court_application", "court_result"}

    hist = history[history["event_date"] <= as_of].copy()

    def _agg(group: pd.DataFrame) -> pd.Series:
        measures_used: Set[str] = set(group["measure"].values)
        had_info = bool(measures_used & informing_measures)
        had_restrict = bool(measures_used & restriction_measures)
        had_court = bool(measures_used & court_measures)

        if had_court:
            stage = "court"
        elif had_restrict:
            stage = "restriction"
        elif "restriction_notice" in measures_used:
            stage = "restriction_notice"
        elif had_info:
            stage = "informing"
        else:
            stage = "unknown"

        return pd.Series({
            "stage": stage,
            "ever_restricted": had_restrict,
            "ever_in_court": had_court,
            "had_informing": had_info,
            "can_do_restriction": had_info,
            "can_do_court": had_restrict,
        })

    if hist.empty:
        return pd.DataFrame(columns=["ls", "stage", "ever_restricted", "ever_in_court",
                                     "had_informing", "can_do_restriction", "can_do_court"])

    state = hist.groupby("ls", as_index=True).apply(_agg).reset_index()
    return state


# ---------------------------------------------------------------------------
# Rules engine: Приложение №1 — eligibility(i, m)
# ---------------------------------------------------------------------------

def check_court_multi_threshold(
    overdue_months: float,
    dz: float,
    thresholds: List[dict],
) -> bool:
    """OR-условие для суда: достаточно выполнения хотя бы одного порога."""
    for t in thresholds:
        if overdue_months >= t.get("min_overdue_months", 0) and dz >= t.get("min_dz", 0):
            return True
    return False


def compute_eligibility(
    features: pd.DataFrame,
    state: pd.DataFrame,
    rules: dict,
) -> pd.DataFrame:
    """
    Строит булеву матрицу eligible[ls, measure].
    Только меры из rules.yaml проверяются; остальные — eligible=True по умолчанию.
    Возвращает DataFrame [ls, measure_name → bool].
    """
    df = features.merge(state, on="ls", how="left")

    # Заполняем дефолты state для ЛС без истории мер
    for col in ["stage", "ever_restricted", "ever_in_court", "had_informing",
                "can_do_restriction", "can_do_court"]:
        if col not in df.columns:
            df[col] = False if col != "stage" else "unknown"
    df["can_do_restriction"] = df["can_do_restriction"].fillna(False)
    df["can_do_court"] = df["can_do_court"].fillna(False)
    df["ever_restricted"] = df["ever_restricted"].fillna(False)
    df["ever_in_court"] = df["ever_in_court"].fillna(False)
    df["had_informing"] = df["had_informing"].fillna(False)
    df["stage"] = df["stage"].fillna("unknown")

    dz = df["DZ_current"].fillna(0).clip(lower=0)

    # overdue_months: используем готовый признак если есть (после rebuild features),
    # иначе — max_consecutive_nonpayment из существующего parquet (хороший прокси).
    if "overdue_months" in df.columns:
        overdue = df["overdue_months"].fillna(0)
    elif "max_consecutive_nonpayment" in df.columns:
        overdue = df["max_consecutive_nonpayment"].fillna(0)
        log.info("overdue_months не найден в признаках, используется max_consecutive_nonpayment как прокси")
    else:
        overdue = pd.Series(0, index=df.index)
        log.warning("overdue_months и max_consecutive_nonpayment отсутствуют — все меры будут eligible по сроку ДЗ")

    has_phone = df["has_phone"].fillna("Unknown") if "has_phone" in df.columns else pd.Series("Unknown", index=df.index)
    has_ebill = df["has_ebill"].fillna("Unknown") if "has_ebill" in df.columns else pd.Series("Unknown", index=df.index)
    avg_accrual = df["avg_accrual_12m"].fillna(0).clip(lower=0) if "avg_accrual_12m" in df.columns else pd.Series(0, index=df.index)

    eligible = pd.DataFrame({"ls": df["ls"]})

    for measure_name in MEASURE_FILES:
        if measure_name not in rules:
            eligible[measure_name] = True
            continue

        r = rules[measure_name]
        mask = pd.Series(True, index=df.index)

        # Срок ДЗ
        min_om = r.get("min_overdue_months")
        max_om = r.get("max_overdue_months")
        if min_om is not None:
            mask &= overdue >= min_om
        if max_om is not None:
            mask &= overdue <= max_om

        # Сумма ДЗ
        min_dz = r.get("min_dz")
        max_dz = r.get("max_dz")
        if min_dz is not None:
            mask &= dz >= min_dz
        if max_dz is not None:
            mask &= dz <= max_dz

        # Контактность
        if r.get("requires_phone") == "Yes":
            mask &= has_phone == "Yes"
        if r.get("requires_email") == "Yes":
            mask &= has_ebill == "Yes"
        if r.get("requires_no_phone"):
            mask &= has_phone != "Yes"

        # Норматив потребления (ДЗ > N × avg_accrual)
        dz_norm = r.get("dz_times_norm_min")
        if dz_norm is not None:
            mask &= dz >= (avg_accrual * dz_norm)

        # Очередность процедур
        prev_stage = r.get("requires_prev_stage")
        if prev_stage == "informing":
            mask &= df["had_informing"]
        elif prev_stage == "restriction_notice":
            mask &= df["can_do_restriction"]
        elif prev_stage == "restriction_or_impossible":
            # Суд: ограничение применено ИЛИ невозможно
            # Невозможность ограничения = is_gasified (газ) или has_benefits
            is_gasified = df["is_gasified"].fillna("No") if "is_gasified" in df.columns else pd.Series("No", index=df.index)
            has_benefits = df["has_benefits"].fillna("No") if "has_benefits" in df.columns else pd.Series("No", index=df.index)
            restriction_impossible = (is_gasified == "Yes") | (has_benefits == "Yes")
            mask &= df["can_do_court"] | restriction_impossible

            # Multi-threshold: OR-условие для суда
            multi = r.get("multi_threshold")
            if multi:
                mt_mask = pd.Series(False, index=df.index)
                for t in multi:
                    mt_mask |= (
                        (overdue >= t.get("min_overdue_months", 0)) &
                        (dz >= t.get("min_dz", 0))
                    )
                mask &= mt_mask

        eligible[measure_name] = mask.values

    return eligible


# ---------------------------------------------------------------------------
# Combo builder: action sets per ЛС
# ---------------------------------------------------------------------------

def build_action_sets(
    features: pd.DataFrame,
    utility: pd.DataFrame,
    eligible: pd.DataFrame,
    cost_tiers: dict,
) -> pd.DataFrame:
    """
    Для каждого ЛС строит лучший action_set:
      - background: E-mail (если has_ebill) + Автодозвон (если has_phone и eligible)
      - leading   : лучшая ведущая мера по expected_recovery_gain среди eligible (cost_tier — tiebreaker)

    Возвращает DataFrame:
      ls, leading_measure, background_measures, all_measures,
      leading_erg, set_erg, set_cost_tier_sum
    """
    df = features[["ls"]].copy()
    df = df.merge(utility, on="ls", how="left")
    df = df.merge(eligible, on="ls", how="left", suffixes=("", "_elig"))
    has_phone = features.set_index("ls")["has_phone"].fillna("Unknown")
    has_ebill = features.set_index("ls")["has_ebill"].fillna("Unknown")

    erg_cols = {m: f"expected_recovery_gain_{m}" for m in LEADING_MEASURES if f"expected_recovery_gain_{m}" in df.columns}

    results = []
    for row in df.itertuples(index=False):
        ls = row.ls
        ls_erg: Dict[str, float] = {}
        ls_elig: Dict[str, bool] = {}

        for m in LEADING_MEASURES:
            erg_col = f"expected_recovery_gain_{m}"
            erg_val = getattr(row, erg_col, 0.0) or 0.0
            elig_val = bool(getattr(row, m, False))
            ls_erg[m] = float(erg_val)
            ls_elig[m] = elig_val

        # Лучшая ведущая мера: max expected_recovery_gain среди eligible,
        # при ничьей — минимальный cost_tier
        candidates = [
            (ls_erg.get(m, 0.0), -cost_tiers.get(m, 3), m)
            for m in LEADING_MEASURES
            if ls_elig.get(m, False) and ls_erg.get(m, 0.0) > 0
        ]
        if candidates:
            best = max(candidates, key=lambda x: (x[0], x[1]))
            leading_measure = best[2]
            leading_erg = best[0]
        else:
            leading_measure = None
            leading_erg = 0.0

        # Background меры
        background: List[str] = []
        phone_ok = has_phone.get(ls, "Unknown") == "Yes"
        ebill_ok = has_ebill.get(ls, "Unknown") == "Yes"
        if ebill_ok and bool(getattr(row, "email", False)):
            background.append("email")
        if phone_ok and bool(getattr(row, "autodial", False)):
            background.append("autodial")

        # Объединяем
        all_measures = list(background)
        if leading_measure:
            all_measures.append(leading_measure)

        # Суммарная ERG и cost_tier_sum
        set_erg = sum(
            float(getattr(row, f"expected_recovery_gain_{m}", 0.0) or 0.0)
            for m in all_measures
            if f"expected_recovery_gain_{m}" in [f"expected_recovery_gain_{x}" for x in ALL_MEASURES]
        )
        set_cost_tier = sum(cost_tiers.get(m, 3) for m in all_measures)

        results.append({
            "ls": ls,
            "leading_measure": leading_measure,
            "background_measures": ";".join(background),
            "all_measures": ";".join(all_measures),
            "leading_erg": round(leading_erg, 2),
            "set_erg": round(set_erg, 2),
            "set_cost_tier_sum": set_cost_tier,
        })

    return pd.DataFrame(results)


def _fast_action_sets(
    features: pd.DataFrame,
    utility: pd.DataFrame,
    eligible: pd.DataFrame,
    cost_tiers: dict,
) -> pd.DataFrame:
    """
    Векторизованная версия build_action_sets (намного быстрее для 100k+ строк).
    """
    base = features[["ls"]].copy()
    has_phone = features.set_index("ls")["has_phone"].fillna("Unknown")
    has_ebill = features.set_index("ls")["has_ebill"].fillna("Unknown")

    # Присоединяем utility и eligibility
    u = utility.set_index("ls")
    e = eligible.set_index("ls")

    lead_candidates = {}
    for m in LEADING_MEASURES:
        erg_col = f"expected_recovery_gain_{m}"
        if erg_col not in u.columns:
            continue
        erg = u[erg_col].fillna(0.0).clip(lower=0)
        elig = e[m].fillna(False) if m in e.columns else pd.Series(False, index=e.index)
        tier = cost_tiers.get(m, 3)
        # score = ERG при elig=True, иначе -inf
        lead_candidates[m] = {
            "erg": erg,
            "elig": elig,
            "tier": tier,
        }

    # Матрица скоров для ведущей меры
    ls_all = base["ls"].values
    leading_measure_arr = np.full(len(ls_all), None, dtype=object)
    leading_erg_arr = np.zeros(len(ls_all))
    leading_tier_arr = np.full(len(ls_all), 99)

    for m, d in lead_candidates.items():
        erg_vals = d["erg"].reindex(ls_all).fillna(0.0).values
        elig_vals = d["elig"].reindex(ls_all).fillna(False).values
        tier = d["tier"]
        # Заменяем лучшего кандидата: больше ERG, при равенстве — меньше tier
        better = (
            elig_vals &
            (erg_vals > 0) &
            (
                (erg_vals > leading_erg_arr) |
                ((erg_vals == leading_erg_arr) & (tier < leading_tier_arr))
            )
        )
        leading_measure_arr = np.where(better, m, leading_measure_arr)
        leading_erg_arr = np.where(better, erg_vals, leading_erg_arr)
        leading_tier_arr = np.where(better, tier, leading_tier_arr)

    # Background
    phone_ok = has_phone.reindex(ls_all).fillna("Unknown").values == "Yes"
    ebill_ok = has_ebill.reindex(ls_all).fillna("Unknown").values == "Yes"

    email_elig = e["email"].reindex(ls_all).fillna(False).values if "email" in e.columns else np.zeros(len(ls_all), bool)
    autodial_elig = e["autodial"].reindex(ls_all).fillna(False).values if "autodial" in e.columns else np.zeros(len(ls_all), bool)

    has_email_bg = ebill_ok & email_elig
    has_autodial_bg = phone_ok & autodial_elig

    results = []
    for i, ls in enumerate(ls_all):
        lm = leading_measure_arr[i]
        bg = []
        if has_email_bg[i]:
            bg.append("email")
        if has_autodial_bg[i]:
            bg.append("autodial")
        all_m = bg + ([lm] if lm else [])

        erg_bg = sum(
            float(u.loc[ls, f"expected_recovery_gain_{m}"] if f"expected_recovery_gain_{m}" in u.columns else 0)
            for m in bg
        ) if ls in u.index else 0.0
        set_erg = round(float(leading_erg_arr[i]) + erg_bg, 2)
        set_tier = sum(cost_tiers.get(m, 3) for m in all_m)

        results.append({
            "ls": ls,
            "leading_measure": lm if lm else None,
            "background_measures": ";".join(bg),
            "all_measures": ";".join(all_m),
            "leading_erg": round(float(leading_erg_arr[i]), 2),
            "set_erg": set_erg,
            "set_cost_tier_sum": set_tier,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Greedy allocator: Приложение №2 — лимиты
# ---------------------------------------------------------------------------

def greedy_allocate(
    action_sets: pd.DataFrame,
    features: pd.DataFrame,
    limits: Dict[str, Optional[int]],
    hot_segment_ids: Tuple[int, ...] = (0, 1),
) -> pd.DataFrame:
    """
    Жадный аллокатор с лимитами Приложения №2.

    Два прохода:
      1) «Горячие» ЛС (сегменты 0, 1 — высокая ДЗ) — в приоритете.
      2) Остальные ЛС по убыванию set_erg.

    Если лимит исчерпан по leading_measure — пробуем назначить без неё
    (только background E-mail/Автодозвон).
    """
    seg_map = features.set_index("ls")["segment"].fillna(-1).astype(int)
    used: Dict[str, int] = {m: 0 for m in limits}

    def fits(measures: List[str]) -> bool:
        for m in measures:
            lim = limits.get(m)
            if lim is not None and used.get(m, 0) >= lim:
                return False
        return True

    def consume(measures: List[str]) -> None:
        for m in measures:
            if m in used:
                used[m] += 1
            else:
                used[m] = 1

    def parse_measures(s: str) -> List[str]:
        return [x for x in (s or "").split(";") if x]

    # Добавим сегмент к action_sets для сортировки
    df = action_sets.copy()
    df["segment"] = df["ls"].map(seg_map).fillna(-1).astype(int)
    df["is_hot"] = df["segment"].isin(hot_segment_ids)

    # Сортируем: сначала горячие по убыванию ERG, потом остальные
    df_hot = df[df["is_hot"]].sort_values("set_erg", ascending=False)
    df_cold = df[~df["is_hot"]].sort_values("set_erg", ascending=False)
    df_sorted = pd.concat([df_hot, df_cold], ignore_index=True)

    assigned_measures = []
    assigned_erg = []

    for row in df_sorted.itertuples(index=False):
        all_m = parse_measures(row.all_measures)
        bg_m = parse_measures(row.background_measures)
        lead = row.leading_measure

        if fits(all_m):
            consume(all_m)
            assigned_measures.append(row.all_measures)
            assigned_erg.append(row.set_erg)
        elif bg_m and fits(bg_m):
            # Ведущая мера превысила лимит — назначаем только фон
            consume(bg_m)
            assigned_measures.append(";".join(bg_m))
            assigned_erg.append(
                row.set_erg - row.leading_erg if row.leading_measure else row.set_erg
            )
        else:
            assigned_measures.append("")
            assigned_erg.append(0.0)

    df_sorted["assigned_measures"] = assigned_measures
    df_sorted["assigned_erg"] = assigned_erg
    df_sorted["assigned"] = df_sorted["assigned_measures"] != ""
    return df_sorted


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------

def generate_explanation(
    ls: int,
    segment: int,
    dz: float,
    overdue_months: int,
    payment_ratio: float,
    measures: List[str],
    leading_measure: Optional[str],
    leading_erg: float,
    leading_tier: int,
    rules: dict,
    triggered_conditions: List[str],
) -> str:
    seg_name = SEGMENT_NAMES.get(segment, f"Сегмент {segment}")
    measures_ru = [MEASURE_RU.get(m, m) for m in measures if m]

    parts = [
        f"Сегмент {segment} — «{seg_name}».",
        f"ДЗ: {dz:,.0f} руб., просрочка: {overdue_months} мес., payment_ratio_12m: {payment_ratio:.2f}.",
    ]

    if leading_measure and str(leading_measure) not in ("", "None", "nan"):
        lead_ru = MEASURE_RU.get(leading_measure, leading_measure)
        tier_desc = {1: "минимальные", 2: "низкие", 3: "умеренные", 4: "высокие", 5: "максимальные"}.get(leading_tier, "")
        parts.append(
            f"Ведущая мера: «{lead_ru}» — "
            f"ожидаемый прирост возврата ≈ {leading_erg:,.0f} руб. "
            f"(cost_tier={leading_tier}, затраты {tier_desc})."
        )
    if measures_ru:
        parts.append("Назначены меры: " + ", ".join(f"«{m}»" for m in measures_ru) + ".")
    else:
        parts.append("В текущем месяце мер не назначено: либо ДЗ не соответствует условиям Приложения №1, "
                     "либо лимиты по всем подходящим мерам исчерпаны.")
    if triggered_conditions:
        parts.append("Условия Приложения №1 выполнены: " + "; ".join(triggered_conditions) + ".")

    return " ".join(parts)


def build_explanations(
    allocated: pd.DataFrame,
    features: pd.DataFrame,
    rules: dict,
    cost_tiers: dict,
) -> pd.Series:
    feat_idx = features.set_index("ls")
    explanations = []

    for row in allocated.itertuples(index=False):
        ls = row.ls
        segment = int(row.segment) if pd.notna(row.segment) else -1
        measures = [x for x in (row.assigned_measures or "").split(";") if x]
        lead = row.leading_measure if hasattr(row, "leading_measure") else None
        lead_erg = float(row.leading_erg) if hasattr(row, "leading_erg") else 0.0
        lead_tier = cost_tiers.get(lead, 3) if lead else 0

        try:
            fr = feat_idx.loc[ls]
            dz = float(fr.get("DZ_current", 0) or 0)
            overdue = int(
                fr.get("overdue_months", None)
                or fr.get("max_consecutive_nonpayment", None)
                or 0
            )
            pay_ratio = float(fr.get("payment_ratio_12m", 0) or 0)
        except (KeyError, TypeError):
            dz, overdue, pay_ratio = 0.0, 0, 0.0

        # Формируем выполненные условия для объяснения
        triggered = []
        if lead and lead in rules:
            r = rules[lead]
            if r.get("min_dz") and dz >= r["min_dz"]:
                triggered.append(f"ДЗ ≥ {r['min_dz']:,} руб.")
            if r.get("min_overdue_months") and overdue >= r["min_overdue_months"]:
                triggered.append(f"просрочка ≥ {r['min_overdue_months']} мес.")
            if r.get("requires_phone") == "Yes":
                triggered.append("есть телефон")
            if r.get("requires_email") == "Yes":
                triggered.append("есть e-mail")
            if r.get("requires_prev_stage"):
                stage_map = {
                    "informing": "информирование применялось",
                    "restriction_notice": "уведомление об ограничении применялось",
                    "restriction_or_impossible": "ограничение применялось или невозможно",
                }
                triggered.append(stage_map.get(r["requires_prev_stage"], r["requires_prev_stage"]))

        expl = generate_explanation(
            ls=ls, segment=segment, dz=dz, overdue_months=overdue,
            payment_ratio=pay_ratio, measures=measures,
            leading_measure=lead, leading_erg=lead_erg,
            leading_tier=lead_tier, rules=rules,
            triggered_conditions=triggered,
        )
        explanations.append(expl)

    return pd.Series(explanations, index=allocated.index)


# ---------------------------------------------------------------------------
# Сборка итогового плана
# ---------------------------------------------------------------------------

def assemble_plan(
    allocated: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    explanations: pd.Series,
    utility: pd.DataFrame,
) -> pd.DataFrame:
    """Объединяет все части в финальный датафрейм рекомендаций."""
    feat_cols = ["ls", "DZ_current", "overdue_months", "max_consecutive_nonpayment",
                 "avg_accrual_12m", "payment_ratio_12m", "has_phone", "has_ebill", "stage_reached"]
    feat_cols = [c for c in feat_cols if c in features.columns]
    feat_sub = features[feat_cols].copy()
    if "overdue_months" not in feat_sub.columns and "max_consecutive_nonpayment" in feat_sub.columns:
        feat_sub["overdue_months"] = feat_sub["max_consecutive_nonpayment"]

    plan = allocated[["ls", "segment", "leading_measure", "background_measures",
                       "assigned_measures", "leading_erg", "assigned_erg",
                       "set_cost_tier_sum", "assigned"]].copy()
    plan = plan.merge(feat_sub, on="ls", how="left")
    plan = plan.merge(labels[["ls", "segment"]].rename(columns={"segment": "segment_kp"}),
                      on="ls", how="left")

    # Имя сегмента
    plan["segment_name"] = plan["segment"].map(SEGMENT_NAMES).fillna("?")

    # Top uplift среди назначенных мер
    uplift_cols = {m: f"uplift_{m}" for m in MEASURE_FILES if f"uplift_{m}" in utility.columns}
    if uplift_cols:
        u_idx = utility.set_index("ls")
        def _top_uplift(row: pd.Series) -> Tuple[str, float]:
            ms = [x for x in (row.get("assigned_measures") or "").split(";") if x]
            best_m, best_u = "", 0.0
            for m in ms:
                col = f"uplift_{m}"
                if col in u_idx.columns and row["ls"] in u_idx.index:
                    val = float(u_idx.loc[row["ls"], col] or 0)
                    if val > best_u:
                        best_m, best_u = m, val
            return best_m, best_u

        plan["top_uplift_measure"] = plan.apply(lambda r: _top_uplift(r)[0], axis=1)
        plan["top_uplift_value"] = plan.apply(lambda r: _top_uplift(r)[1], axis=1)
    else:
        plan["top_uplift_measure"] = ""
        plan["top_uplift_value"] = 0.0

    plan["explanation"] = explanations.values

    # Переименовываем и упорядочиваем колонки
    plan = plan.rename(columns={
        "assigned_measures": "measures",
        "assigned_erg": "set_expected_recovery_gain",
        "set_cost_tier_sum": "set_cost_tier",
    })

    final_cols = [
        "ls", "segment", "segment_name", "DZ_current", "overdue_months",
        "avg_accrual_12m", "payment_ratio_12m", "has_phone", "has_ebill",
        "stage_reached", "measures", "leading_measure", "background_measures",
        "leading_erg", "set_expected_recovery_gain", "set_cost_tier",
        "top_uplift_measure", "top_uplift_value", "assigned", "explanation",
    ]
    final_cols = [c for c in final_cols if c in plan.columns]
    return plan[final_cols].sort_values("ls").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Сводная статистика
# ---------------------------------------------------------------------------

def build_summary(plan: pd.DataFrame, limits: Dict[str, Optional[int]]) -> pd.DataFrame:
    """Агрегаты по мерам: coverage, utilization лимитов."""
    # Считаем точные вхождения через разбивку строки мер
    measure_counts: Dict[str, int] = {m: 0 for m in MEASURE_FILES}
    for measures_str in plan["measures"].fillna(""):
        for m in [x for x in measures_str.split(";") if x]:
            if m in measure_counts:
                measure_counts[m] += 1

    records = []
    for m in MEASURE_FILES:
        assigned = measure_counts.get(m, 0)
        lim = limits.get(m)
        utilization = f"{100*assigned/lim:.1f}%" if lim else "N/A (no limit)"
        records.append({
            "measure": m,
            "measure_ru": MEASURE_RU.get(m, m),
            "assigned_count": int(assigned),
            "limit": lim if lim else "∞",
            "utilization": utilization,
        })
    summary_measures = pd.DataFrame(records)

    # По сегментам
    seg_summary = (
        plan.groupby("segment")
        .agg(
            n_ls=("ls", "count"),
            n_assigned=("assigned", "sum"),
            mean_dz=("DZ_current", "mean"),
            mean_erg=("set_expected_recovery_gain", "mean"),
        )
        .reset_index()
    )
    seg_summary["segment_name"] = seg_summary["segment"].map(SEGMENT_NAMES)
    seg_summary["coverage_pct"] = (seg_summary["n_assigned"] / seg_summary["n_ls"] * 100).round(1)

    return summary_measures, seg_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Построение персонального плана мер на месяц (п.2.3 + п.3 + п.4)"
    )
    p.add_argument("--data-dir", default="Материалы к кейсу",
                   help="Папка с исходными xlsx/csv файлами")
    p.add_argument("--features", default="output/kprototypes_features.parquet",
                   help="Признаки ЛС (parquet или csv)")
    p.add_argument("--utility", default="output/utility_matrix.parquet",
                   help="Матрица uplift/expected_recovery_gain")
    p.add_argument("--labels", default="output/kprototypes_labels.csv",
                   help="Метки сегментации")
    p.add_argument("--config", default="configs/",
                   help="Папка с конфигами (limits.yaml, rules.yaml, costs.yaml)")
    p.add_argument("--month", default=None,
                   help="Месяц рекомендаций (YYYY-MM), по умолчанию — текущий")
    p.add_argument("--out", default=None,
                   help="Путь к выходному CSV (по умолчанию output/recommendations_YYYY-MM.csv)")
    p.add_argument("--hot-segments", default="0,1",
                   help="Сегменты с приоритетом (через запятую, default: 0,1)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Основной пайплайн
# ---------------------------------------------------------------------------

def run(
    data_dir: Path,
    features_path: Path,
    utility_path: Path,
    labels_path: Path,
    config_dir: Path,
    month: Optional[str],
    out_path: Path,
    hot_segments: Tuple[int, ...],
) -> None:
    # Определяем период
    if month:
        as_of = pd.Timestamp(month + "-01") + pd.offsets.MonthEnd(0)
    else:
        as_of = pd.Timestamp.now().normalize()
    month_str = as_of.strftime("%Y-%m")
    log.info("Месяц рекомендаций: %s (as_of=%s)", month_str, as_of.date())

    # Загрузка конфигов
    log.info("Загрузка конфигов: %s", config_dir)
    rules, limits, costs_cfg = load_configs(config_dir)
    cost_tiers = costs_cfg["tiers"]
    log.info("Лимиты: %s", {k: v for k, v in limits.items() if v is not None})

    # Загрузка данных
    log.info("Загрузка признаков: %s", features_path)
    features = load_features(features_path)
    log.info("Загрузка utility matrix: %s", utility_path)
    utility = load_utility(utility_path)
    log.info("Загрузка меток сегментации: %s", labels_path)
    labels = load_labels(labels_path)

    # Добавляем сегмент в features
    features = features.merge(labels, on="ls", how="left")
    features["segment"] = features["segment"].fillna(-1).astype(int)
    log.info("Загружено %d ЛС", len(features))

    # История мер — для state machine
    log.info("Загрузка истории мер: %s", data_dir)
    history = load_measure_history(data_dir)
    log.info("Событий мер: %d", len(history))

    # State machine
    log.info("Вычисляем state machine (стадия процедуры) ...")
    state = compute_procedure_state(history, as_of=as_of)
    log.info("State machine: %d ЛС с историей мер", len(state))

    # Eligibility
    log.info("Применяем правила Приложения №1 (eligible[i,m]) ...")
    eligible = compute_eligibility(features, state, rules)
    n_eligible = {m: int(eligible[m].sum()) for m in MEASURE_FILES if m in eligible.columns}
    log.info("Eligible ЛС по мерам: %s", n_eligible)

    # Combo builder
    log.info("Строим action_sets (фон + ведущая мера) ...")
    action_sets = _fast_action_sets(features, utility, eligible, cost_tiers)
    n_with_lead = (action_sets["leading_measure"].notna()).sum()
    log.info("ЛС с ведущей мерой: %d / %d", n_with_lead, len(action_sets))

    # Greedy allocator
    log.info("Жадный аллокатор с лимитами Приложения №2 ...")
    allocated = greedy_allocate(action_sets, features, limits, hot_segments)
    n_assigned = int(allocated["assigned"].sum())
    log.info("Назначено: %d ЛС из %d (%.1f%%)", n_assigned, len(allocated), 100*n_assigned/len(allocated))

    # Объяснения
    log.info("Генерируем объяснения ...")
    explanations = build_explanations(allocated, features, rules, cost_tiers)

    # Сборка финального плана
    plan = assemble_plan(allocated, features, labels, explanations, utility)

    # Сохраняем
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info("Рекомендации сохранены: %s (%d строк)", out_path, len(plan))

    # Сводка
    summary_measures, seg_summary = build_summary(plan, limits)
    summary_path = out_path.parent / "recommendations_summary.csv"
    summary_measures.to_csv(summary_path, index=False, encoding="utf-8-sig")
    log.info("\nSummary по мерам:\n%s", summary_measures.to_string(index=False))
    log.info("\nSummary по сегментам:\n%s", seg_summary.to_string(index=False))
    log.info("\nГотово! Файлы:\n  %s\n  %s", out_path, summary_path)


def main() -> None:
    args = parse_args()
    base = Path(__file__).parent.parent

    if args.month:
        month_tag = args.month
    else:
        month_tag = pd.Timestamp.now().strftime("%Y-%m")

    out_path = (
        Path(args.out)
        if args.out
        else base / f"output/recommendations_{month_tag}.csv"
    )

    hot_segs = tuple(int(x.strip()) for x in args.hot_segments.split(",") if x.strip())

    run(
        data_dir=base / args.data_dir,
        features_path=base / args.features,
        utility_path=base / args.utility,
        labels_path=base / args.labels,
        config_dir=base / args.config,
        month=args.month,
        out_path=out_path,
        hot_segments=hot_segs,
    )


if __name__ == "__main__":
    main()
