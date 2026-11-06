"""Ручная сверка срока просрочки (DATA-5, P-02): пошаговый FIFO по очереди счетов.

В панели (tns/data/panel.py) срок считается коротким путём, по сальдо. Здесь тот же срок
считается так, как его считал бы человек с карандашом: ведём список счетов по месяцам и для
каждого помним, сколько по нему ещё не оплачено.

- Новый счёт (начислено > 0) встаёт в конец очереди.
- Всё, что уменьшает долг (оплата, отрицательный перерасчёт, корректировка сальдо вниз),
  гасит самые старые счета (FIFO).
- Всё, что увеличивает долг, кроме нового счёта (возврат оплаты, корректировка сальдо вверх),
  снова открывает самые свежие оплаченные счета: возвращаются последние деньги.
- Переплата копится и гасит следующие счета.
- Корректировка — разница между сальдо следующего месяца и «сальдо + начислено − оплачено».

На 01.M счёт за M−1 ещё не просрочен (оплата до 15.M). Просроченные — непогашенные счета за M−2
и раньше; срок — сколько месяцев от самого старого из них до M−2 включительно. Долг, который
был до начала данных, — отдельный «счёт» без даты: срок по нему занижен (censored).

Запуск: python -m tns.data.overdue_check --month 2026-03 [--ls 9044 11988 ...]
Читает output/<ГГГГ-ММ>/panel.parquet и пишет туда же overdue_check.xlsx: по 10 ЛС помесячно
очередь счетов, срок по ручному расчёту и срок из панели.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tns.logs import report_error, setup_logging
from tns.data.loaders import REPO_ROOT, load_settings
from tns.data.panel import KOPECK, parse_plan_month

logger = logging.getLogger(__name__)

BEFORE_DATA = -1  # «счёт» за долг, который был до начала данных


@dataclass
class Bill:
    period: int        # номер месяца данных, BEFORE_DATA — долг до начала данных
    amount: float      # сумма счёта
    unpaid: float      # сколько ещё не оплачено


@dataclass
class MonthState:
    """Состояние ЛС на 01.M по ручному расчёту."""
    open_bills: list[tuple[int, float]]   # непогашенные счета (период, остаток)
    dz_overdue: float                      # просроченная часть, NaN — не определить
    overdue_m: float                       # срок просрочки, NaN — не определить
    censored: bool                         # долг старше данных


def _pay_oldest(bills: list[Bill], amount: float) -> float:
    """Гасит самые старые счета, возвращает остаток (переплату)."""
    for bill in bills:
        if amount <= 0:
            break
        take = min(bill.unpaid, amount)
        bill.unpaid -= take
        amount -= take
    return amount


def _reopen_newest(bills: list[Bill], amount: float) -> None:
    """Снова открывает самые свежие оплаченные счета. Что не поместилось — долг до начала данных."""
    for bill in reversed(bills):
        if amount <= 0:
            return
        take = min(bill.amount - bill.unpaid, amount)
        bill.unpaid += take
        amount -= take
    if amount > 0:
        if bills and bills[0].period == BEFORE_DATA:
            bills[0].amount += amount
            bills[0].unpaid += amount
        else:
            bills.insert(0, Bill(BEFORE_DATA, amount, amount))


def fifo_by_bills(saldo_start, accrual, paid, zero_accrual_months: str = "count") -> list[MonthState]:
    """Ручной расчёт срока для одного ЛС. Аргументы — значения по месяцам подряд с начала данных."""
    if zero_accrual_months not in {"count", "skip"}:
        raise ValueError(f"Неизвестное значение overdue.zero_accrual_months: «{zero_accrual_months}». "
                         "Можно: count или skip. Исправьте это в configs/settings.yaml.")
    saldo = np.asarray(saldo_start, dtype=float)
    charged = np.nan_to_num(np.asarray(accrual, dtype=float))
    payment = np.nan_to_num(np.asarray(paid, dtype=float))
    n = len(saldo)

    bills = [Bill(BEFORE_DATA, saldo[0], saldo[0])] if saldo[0] > 0 else []
    credit = max(0.0, -saldo[0])  # переплата
    states = []
    for t in range(n):
        open_bills = [(b.period, round(b.unpaid, 2)) for b in bills if b.unpaid >= KOPECK]
        if np.isnan(saldo[t]):
            states.append(MonthState(open_bills, np.nan, np.nan, False))
        elif t == 0:
            # Начисление за декабрь 2024 неизвестно: какая часть долга просрочена, не определить.
            debt = saldo[0] > 0
            states.append(MonthState(open_bills, np.nan if debt else 0.0, np.nan if debt else 0, debt))
        else:
            overdue = [(k, r) for k, r in open_bills if k <= t - 2]
            if not overdue:
                states.append(MonthState(open_bills, 0.0, 0, False))
            else:
                oldest = overdue[0][0]
                if zero_accrual_months == "count":
                    months = (t - 1) - oldest
                else:
                    months = int((charged[max(oldest, 0):t - 1] > 0).sum()) + (oldest == BEFORE_DATA)
                states.append(MonthState(open_bills, round(sum(r for _, r in overdue), 2), months, oldest == BEFORE_DATA))

        if t == n - 1:
            break
        # Движение за месяц t.
        if charged[t] > 0:
            bills.append(Bill(t, charged[t], charged[t]))
        expected = saldo[t] + charged[t] - payment[t]
        correction = 0.0 if np.isnan(saldo[t + 1]) or np.isnan(saldo[t]) else saldo[t + 1] - expected
        down = max(payment[t], 0) + max(-charged[t], 0) + max(-correction, 0)
        up = max(-payment[t], 0) + max(correction, 0)
        # Сначала гасим переплатой и уменьшениями, потом открываем заново: так остаток долга
        # всегда состоит из самых свежих счетов, как и в расчёте по сальдо.
        credit = _pay_oldest(bills, credit + down)
        if up > 0:
            used = min(credit, up)
            credit -= used
            _reopen_newest(bills, up - used)
    return states


# ---------------------------------------------------------------------------
# Выбор ЛС и выгрузка для ручной сверки
# ---------------------------------------------------------------------------

def _matrices(panel: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex, dict[str, np.ndarray]]:
    panel = panel.sort_values(["ls", "month"])
    months = pd.DatetimeIndex(sorted(panel["month"].unique()))
    ls_values = np.sort(panel["ls"].unique())
    row = np.searchsorted(ls_values, panel["ls"].to_numpy())
    col = months.get_indexer(panel["month"])
    out = {}
    for name in ["saldo_start", "accrual", "paid", "dz", "dz_overdue", "overdue_m"]:
        matrix = np.full((len(ls_values), len(months)), np.nan)
        matrix[row, col] = panel[name].astype("float64").to_numpy()
        out[name] = matrix
    return ls_values, months, out


def pick_cases(panel: pd.DataFrame) -> dict[str, int]:
    """10 ЛС с разными ситуациями: для каждой — ЛС с наименьшим номером, у которого она есть."""
    ls, _, m = _matrices(panel)
    s, a, p, dz, om = m["saldo_start"], m["accrual"], m["paid"], m["dz"], m["overdue_m"]
    gap = s[:, 1:] - (s[:, :-1] + np.nan_to_num(a[:, :-1]) - np.nan_to_num(p[:, :-1]))
    reconciles = (np.abs(gap) <= 0.01).all(axis=1)
    uncensored_start = s[:, 0] <= 0
    rules = {
        "долг растёт без оплат": uncensored_start & (np.nanmax(om, axis=1) >= 6) & reconciles,
        "частичные оплаты": reconciles & uncensored_start & ((p > 0) & (p < a)).sum(axis=1).__ge__(3) & (np.nanmax(om, axis=1) >= 3),
        "погасил долг, потом новый": reconciles & (np.nanmax(om[:, :7], axis=1) >= 2) & (om[:, 7:10] == 0).any(axis=1) & (np.nanmax(om[:, 10:], axis=1) >= 2),
        "переплата гасит новые счета": reconciles & (s < -500).any(axis=1) & (np.nanmax(om, axis=1) >= 2),
        "возврат оплаты": reconciles & (p < -100).any(axis=1) & (np.nanmax(om, axis=1) >= 1),
        "отрицательный перерасчёт": reconciles & (a < -100).any(axis=1) & (np.nanmax(om, axis=1) >= 2),
        "месяц без начисления при долге": reconciles & ((a[:, 1:-1] == 0) & (dz[:, 1:-1] > 500)).any(axis=1) & (np.nanmax(om, axis=1) >= 3),
        "долг старше начала данных": reconciles & (s[:, 0] > 5000) & (dz[:, -1] > 5000),
        "сальдо выросло без начисления": (gap > 100).any(axis=1) & (np.nanmax(om, axis=1) >= 2),
        "сальдо уменьшилось без оплаты": (gap < -100).any(axis=1) & (np.nanmax(om, axis=1) >= 2),
    }
    cases, used = {}, set()
    for name, mask in rules.items():
        candidates = [int(x) for x in ls[mask] if int(x) not in used]
        if candidates:
            cases[name] = candidates[0]
            used.add(candidates[0])
        else:
            logger.warning("Сверка: не нашлось ЛС для ситуации «%s».", name)
    return cases


def _period_name(k: int, months: pd.DatetimeIndex) -> str:
    return "до начала данных" if k == BEFORE_DATA else f"{months[k]:%Y-%m}"


def check_table(panel: pd.DataFrame, cases: dict[str, int], zero_accrual_months: str = "count") -> pd.DataFrame:
    """Помесячная таблица для ручной сверки: очередь счетов, ручной срок, срок из панели."""
    ls, months, m = _matrices(panel[panel["ls"].isin(list(cases.values()))])
    rows = []
    for situation, account in cases.items():
        i = int(np.searchsorted(ls, account))
        states = fifo_by_bills(m["saldo_start"][i], m["accrual"][i], m["paid"][i], zero_accrual_months)
        for t, state in enumerate(states):
            nxt = m["saldo_start"][i, t + 1] if t + 1 < len(months) else np.nan
            correction = nxt - (m["saldo_start"][i, t] + np.nan_to_num(m["accrual"][i, t]) - np.nan_to_num(m["paid"][i, t]))
            panel_m = m["overdue_m"][i, t]
            rows.append({
                "Ситуация": situation,
                "ЛС": account,
                "Месяц (на 01-е)": f"{months[t]:%Y-%m}",
                "Сальдо на 01-е": m["saldo_start"][i, t],
                "Непогашенные счета на 01-е": "; ".join(f"{_period_name(k, months)}: {r:.2f}" for k, r in state.open_bills) or "нет",
                "Просрочено, руб. (ручной)": state.dz_overdue,
                "Срок, мес. (ручной)": state.overdue_m,
                "Срок, мес. (панель)": panel_m,
                "Совпало": (np.isnan(panel_m) and np.isnan(state.overdue_m)) or panel_m == state.overdue_m,
                "Долг старше данных": state.censored,
                "Начислено за месяц": m["accrual"][i, t],
                "Оплачено за месяц": m["paid"][i, t],
                "Корректировка сальдо": round(correction, 2) if abs(correction) > 0.01 else 0.0,
            })
    return pd.DataFrame(rows)


RULES_TEXT = [
    "Как читать сверку срока просрочки (P-02, FIFO)",
    "",
    "Каждый месяц — один счёт на сумму «Начислено». В колонке «Непогашенные счета» — какие счета на 01-е ещё не оплачены и на сколько.",
    "Оплата, отрицательный перерасчёт и корректировка сальдо вниз гасят самые старые счета.",
    "Возврат оплаты и корректировка сальдо вверх снова открывают самые свежие оплаченные счета.",
    "Корректировка = сальдо следующего месяца − (сальдо + начислено − оплачено).",
    "На 01.M счёт за M−1 ещё не просрочен (оплата до 15.M). Срок = сколько месяцев от самого старого непогашенного счёта до M−2 включительно.",
    "Долг до начала данных (2025-01) — счёт без даты: срок по нему занижен («Долг старше данных» = Да).",
    "В первом месяце данных срок не определить: неизвестно начисление за декабрь 2024.",
    "«Срок (ручной)» посчитан по очереди счетов (tns/data/overdue_check.py), «Срок (панель)» — из panel.parquet. Проверьте вручную 1–2 месяца по каждому ЛС.",
]


def write_check(table: pd.DataFrame, path: Path) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"": RULES_TEXT}).to_excel(writer, sheet_name="Как читать", index=False, header=False)
        table.to_excel(writer, sheet_name="Сверка", index=False)
        sheet = writer.sheets["Сверка"]
        for column, width in zip("ABCDEFGHIJKLM", [30, 10, 14, 14, 60, 14, 12, 12, 10, 12, 14, 14, 14]):
            sheet.column_dimensions[column].width = width
        sheet.freeze_panes = "D2"
        writer.sheets["Как читать"].column_dimensions["A"].width = 140


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ручная сверка срока просрочки (P-02) на 10 ЛС.")
    parser.add_argument("--config", help="файл настроек (по умолчанию configs/settings.yaml)")
    parser.add_argument("--out", help="папка с результатами (по умолчанию из настроек)")
    parser.add_argument("--month", required=True, help="месяц плана ГГГГ-ММ: папка, где лежит panel.parquet")
    parser.add_argument("--ls", nargs="+", type=int, help="номера ЛС (по умолчанию 10 ЛС с разными ситуациями)")
    args = parser.parse_args(argv)
    log_path = setup_logging("overdue_check", argv=argv)

    try:
        settings = load_settings(args.config)
        folder = Path(args.out or REPO_ROOT / settings["paths"]["output_dir"]) / f"{parse_plan_month(args.month):%Y-%m}"
        panel_path = folder / "panel.parquet"
        if not panel_path.exists():
            raise FileNotFoundError(f"Нет файла «{panel_path}». Сначала постройте панель: python -m tns.data.panel --month {args.month}")
        panel = pd.read_parquet(panel_path)
        if args.ls:
            missing = sorted(set(args.ls) - set(panel["ls"]))
            if missing:
                raise ValueError(f"В панели нет ЛС: {', '.join(map(str, missing))}. Проверьте номера в --ls.")
            cases = {f"ЛС {x}": x for x in args.ls}
        else:
            cases = pick_cases(panel)
        table = check_table(panel, cases, settings["overdue"]["zero_accrual_months"])
        write_check(table, folder / "overdue_check.xlsx")
    except Exception as error:  # noqa: BLE001 — любую ошибку объясняем пользователю
        report_error(error, log_path, "Сверка не построена: ")
        return 1

    bad = table.loc[~table["Совпало"], ["ЛС", "Месяц (на 01-е)"]]
    logger.info("Сверка: %d ЛС, %d месяцев, расхождений: %d.", len(cases), len(table), len(bad))
    for _, row in bad.iterrows():
        logger.warning("Расхождение: ЛС %s, %s.", row["ЛС"], row["Месяц (на 01-е)"])
    logger.info("Результат записан в «%s».", folder / "overdue_check.xlsx")
    return 0 if bad.empty else 2


if __name__ == "__main__":
    sys.exit(main())
