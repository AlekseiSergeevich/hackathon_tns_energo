"""Таблицы данных для остальных частей: ls_info, panel, measure_events.

Форматы и смысл колонок — в docs/interfaces.md (разделы 1–3). Исходные файлы читает
tns/data/loaders.py, здесь только расчёты.

Запуск:
    python -m tns.data.panel [--data ПАПКА] [--out ПАПКА] [--month YYYY-MM] [--config ФАЙЛ] [--no-cache]

Колонки panel с known_at = start считаются только по сальдо на 01.M и по данным месяцев
до M. Колонки end (accrual, paid, paid_payments, pay30_m) содержат будущее относительно
среза M и в признаки месяца M попадать не должны.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tns.data.loaders import MEASURE_FILES, REPO_ROOT, default_cache_dir, load_all, load_settings

logger = logging.getLogger(__name__)

WINDOW_MONTH_ENDS = {"month_end": 1, "next_month_end": 2}
KOPECK = 0.005  # суммы меньше полкопейки считаем нулём: это погрешность float

LS_INFO_COLUMNS = [
    "ls", "has_phone", "has_ebill", "has_benefits", "not_residing",
    "remote_disconnect", "house_type", "gasified", "address_guid",
]
PANEL_COLUMNS = [
    "ls", "month", "saldo_start", "dz", "dz_overdue", "overdue_m", "overdue_censored",
    "months_no_payment", "months_no_payment_censored", "norm", "norm_source",
    "accrual", "paid", "paid_payments", "pay30_m",
]
EVENT_COLUMNS = [
    "ls", "measure", "month", "event_date", "date_precision", "n_raw",
    "dz_at_event", "window_end", "paid_in_window", "result_30pct",
]


def _month_start(dates: pd.Series) -> pd.Series:
    return dates.dt.to_period("M").dt.to_timestamp().astype("datetime64[ns]")


# ---------------------------------------------------------------------------
# ls_info
# ---------------------------------------------------------------------------

def build_ls_info(info: pd.DataFrame, osv_ls: pd.Series) -> pd.DataFrame:
    """Анкета на все ЛС из файлов 01 и 02. ЛС только из 02 получают пустые флаги."""
    only_in_osv = np.setdiff1d(osv_ls.unique(), info["ls"].to_numpy())
    only_in_info = np.setdiff1d(info["ls"].to_numpy(), osv_ls.unique())
    if len(only_in_osv) or len(only_in_info):
        logger.warning("ЛС только в файле 02: %d, только в файле 01: %d.", len(only_in_osv), len(only_in_info))
    missing = pd.DataFrame({"ls": only_in_osv.astype("int64")})
    # concat с пустыми строками сохраняет типы колонок анкеты (boolean, category, string).
    result = pd.concat([info, missing.reindex(columns=info.columns).astype(info.dtypes)], ignore_index=True)
    return result.sort_values("ls", ignore_index=True)[LS_INFO_COLUMNS]


# ---------------------------------------------------------------------------
# panel
# ---------------------------------------------------------------------------

def _reached_share(paid: pd.Series, base: pd.Series, share: float) -> pd.Series:
    """paid ≥ share × base. Если долга нет или он неизвестен — <NA>."""
    threshold = (share * base).round(2)
    return (paid >= threshold).astype("boolean").mask(~(base > 0), pd.NA)


def _overdue_from_saldo(dz: np.ndarray, accrual: np.ndarray, zero_accrual_months: str):
    """Просроченная часть долга и срок просрочки по P-02 на 01.M (матрицы ЛС × месяц).

    При FIFO остаток долга всегда состоит из самых свежих начислений, поэтому срок считается
    по сальдо. Счёт за M−1 оплачивается до 15.M и на 01.M ещё не просрочен. Остальной долг
    покрываем начислениями M−2, M−3, … и смотрим, сколько месяцев на это понадобилось.

    Если начислений с начала данных не хватает, долг старше данных: срок занижен, ставим флаг
    censored. В первом месяце данных начисление M−1 неизвестно, поэтому при долге > 0
    dz_overdue и срок не определены (NaN), а censored = True.

    Возвращает float-матрицы dz_overdue и overdue_m и bool-матрицу censored.
    """
    if zero_accrual_months not in {"count", "skip"}:
        raise ValueError(f"Неизвестное значение overdue.zero_accrual_months: «{zero_accrual_months}».")
    # Отрицательные начисления (перерасчёты) не создают долг, который можно покрыть.
    charged = np.clip(np.nan_to_num(accrual), 0, None)
    n_ls, n_months = dz.shape
    dz_overdue = np.zeros((n_ls, n_months))
    overdue_m = np.zeros((n_ls, n_months))
    censored = np.zeros((n_ls, n_months), dtype=bool)

    first_month_debt = dz[:, 0] > 0
    dz_overdue[first_month_debt, 0] = np.nan
    overdue_m[first_month_debt, 0] = np.nan
    censored[first_month_debt, 0] = True

    for t in range(1, n_months):
        remainder = np.nan_to_num(dz[:, t]) - charged[:, t - 1]
        remainder[remainder < KOPECK] = 0.0
        dz_overdue[:, t] = remainder.round(2)
        active = remainder > 0
        for k in range(t - 2, -1, -1):
            if not active.any():
                break
            overdue_m[active, t] += 1 if zero_accrual_months == "count" else charged[active, k] > 0
            remainder[active] -= charged[active, k]
            active = remainder >= KOPECK
        # Начислений не хватило: остаток старше начала данных, это ещё минимум один месяц.
        overdue_m[active, t] += 1
        censored[active, t] = True

    unknown_dz = np.isnan(dz)
    dz_overdue[unknown_dz] = np.nan
    overdue_m[unknown_dz] = np.nan
    return dz_overdue, overdue_m, censored


def _months_without_payment(payments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Сколько месяцев подряд перед M без оплат. Оплата = сумма за месяц > 0 (сторно её гасит)."""
    has_payment = payments > 0
    run = np.zeros(payments.shape, dtype="int64")
    censored = np.zeros(payments.shape, dtype=bool)
    censored[:, 0] = True  # до первого месяца данных истории нет
    for t in range(1, payments.shape[1]):
        paid_before = has_payment[:, t - 1]
        run[:, t] = np.where(paid_before, 0, run[:, t - 1] + 1)
        censored[:, t] = ~paid_before & censored[:, t - 1]
    return run, censored


def _proxy_norm(accrual: np.ndarray, window: int, min_months: int) -> np.ndarray:
    """Прокси норматива: среднее начисление за доступные месяцы ≤ M−1 (до window, минимум min_months)."""
    norm = np.full(accrual.shape, np.nan)
    known = ~np.isnan(accrual)
    values = np.nan_to_num(accrual)
    for t in range(accrual.shape[1]):
        start = max(0, t - window)
        count = known[:, start:t].sum(axis=1)
        enough = count >= min_months
        norm[enough, t] = values[enough, start:t].sum(axis=1) / count[enough]
    return norm


def build_panel(osv: pd.DataFrame, payments: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Панель «ЛС × месяц» из длинной ОСВ (read_osv) и оплат (read_payments)."""
    if settings["overdue"]["allocation"] != "fifo":
        raise ValueError(f"Неизвестное значение overdue.allocation: «{settings['overdue']['allocation']}». Сейчас поддерживается только fifo.")

    panel = osv[["ls", "month", "saldo_start", "accrual", "paid"]].sort_values(["ls", "month"], ignore_index=True)
    panel = panel.astype({"saldo_start": "float64", "accrual": "float64", "paid": "float64"})
    ls_values = np.sort(panel["ls"].unique())
    months = pd.date_range(panel["month"].min(), panel["month"].max(), freq="MS")
    row = np.searchsorted(ls_values, panel["ls"].to_numpy())
    col = months.get_indexer(panel["month"])
    shape = (len(ls_values), len(months))

    def to_matrix(values: pd.Series) -> np.ndarray:
        matrix = np.full(shape, np.nan)
        matrix[row, col] = values.to_numpy("float64")
        return matrix

    pay_month = _month_start(payments["payment_date"])
    in_panel = payments["ls"].isin(ls_values) & pay_month.isin(months)
    if not in_panel.all():
        logger.warning(
            "Оплаты: %d строк вне панели (ЛС нет в ОСВ: %d ЛС; дата вне периода ОСВ: %d строк), в расчёт не вошли.",
            int((~in_panel).sum()),
            payments.loc[~payments["ls"].isin(ls_values), "ls"].nunique(),
            int((~pay_month.isin(months)).sum()),
        )
    monthly = pd.DataFrame({"ls": payments["ls"], "month": pay_month, "amount": payments["amount"]})[in_panel]
    monthly = monthly.groupby(["ls", "month"], as_index=False)["amount"].sum()
    payments_matrix = np.zeros(shape)
    payments_matrix[np.searchsorted(ls_values, monthly["ls"].to_numpy()), months.get_indexer(monthly["month"])] = monthly["amount"].to_numpy()

    panel["dz"] = panel["saldo_start"].clip(lower=0)
    dz_overdue, overdue_m, overdue_censored = _overdue_from_saldo(
        to_matrix(panel["dz"]), to_matrix(panel["accrual"]), settings["overdue"]["zero_accrual_months"]
    )
    no_payment, no_payment_censored = _months_without_payment(payments_matrix)
    norm = _proxy_norm(to_matrix(panel["accrual"]), settings["norm"]["proxy_months"], settings["norm"]["proxy_min_months"])

    panel["dz_overdue"] = dz_overdue[row, col]
    panel["overdue_m"] = pd.Series(overdue_m[row, col], index=panel.index).astype("Int64")
    panel["overdue_censored"] = pd.Series(overdue_censored[row, col], index=panel.index).astype("boolean").mask(panel["dz"].isna(), pd.NA)
    panel["months_no_payment"] = pd.Series(no_payment[row, col], index=panel.index).astype("Int64")
    panel["months_no_payment_censored"] = pd.Series(no_payment_censored[row, col], index=panel.index).astype("boolean")
    panel["norm"] = norm[row, col]
    # Выгрузки заказчика пока нет. Когда появится, её значения получат метку customer.
    panel["norm_source"] = pd.Categorical(np.where(np.isnan(panel["norm"]), None, "proxy"), categories=["customer", "proxy"])
    panel["paid_payments"] = payments_matrix[row, col]
    panel["pay30_m"] = _reached_share(panel["paid_payments"], panel["dz"], settings["result"]["share"])
    panel["month"] = panel["month"].astype("datetime64[ns]")
    return panel[PANEL_COLUMNS]


# ---------------------------------------------------------------------------
# measure_events
# ---------------------------------------------------------------------------

def build_measure_events(events: pd.DataFrame, panel: pd.DataFrame, payments: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Журнал мер: не больше одной строки на ЛС × мера × месяц, с результатом меры."""
    window = settings["result"]["window"]
    if window not in WINDOW_MONTH_ENDS:
        raise ValueError(f"Неизвестное значение result.window: «{window}».")

    grouped = events.assign(month=_month_start(events["event_date"])).groupby(
        ["ls", "measure", "month"], as_index=False, observed=True
    )
    log = grouped.agg(
        event_date=("event_date", "min"), date_precision=("date_precision", "first"), n_raw=("event_date", "size")
    )
    collapsed = log.loc[log["n_raw"] > 1].groupby("measure", observed=True)["n_raw"].apply(lambda n: int((n - 1).sum()))
    if len(collapsed):
        logger.info("Журнал мер: свёрнуто повторов в месяце: %s.", collapsed.to_dict())

    log["n_raw"] = log["n_raw"].astype("int64")
    log["date_precision"] = log["date_precision"].astype(events["date_precision"].dtype)
    log["measure"] = pd.Categorical(log["measure"], categories=list(MEASURE_FILES))
    log = log.merge(
        panel[["ls", "month", "dz"]].rename(columns={"dz": "dz_at_event"}), on=["ls", "month"], how="left"
    )
    outside = log["dz_at_event"].isna()
    if outside.any():
        logger.warning("Журнал мер: %d строк без долга на месяц меры (ЛС или месяца нет в панели).", int(outside.sum()))
    log["window_end"] = (log["month"] + pd.offsets.MonthEnd(WINDOW_MONTH_ENDS[window])).astype("datetime64[ns]")

    # Для мер с точностью «месяц» точной даты нет, поэтому окно открывается с 1-го числа.
    is_month = log["date_precision"] == "month"
    starts = log["event_date"].where(~is_month, log["month"])
    windows = pd.DataFrame({"event": log.index, "ls": log["ls"], "start": starts, "end": log["window_end"]})
    matched = windows.merge(payments[["ls", "payment_date", "amount"]], on="ls", how="inner")
    matched = matched[(matched["payment_date"] >= matched["start"]) & (matched["payment_date"] <= matched["end"])]
    log["paid_in_window"] = matched.groupby("event")["amount"].sum().reindex(log.index, fill_value=0.0).astype("float64")

    log["result_30pct"] = _reached_share(log["paid_in_window"], log["dz_at_event"], settings["result"]["share"])
    log["month"] = log["month"].astype("datetime64[ns]")
    log["event_date"] = log["event_date"].astype("datetime64[ns]")
    return log.sort_values(["ls", "measure", "month"], ignore_index=True)[EVENT_COLUMNS]


# ---------------------------------------------------------------------------
# Сборка и запись
# ---------------------------------------------------------------------------

def build_tables(data_dir: Path | str, settings: dict, cache_dir: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Читает файлы 01–14 (load_all) и строит ls_info, panel и measure_events."""
    logger.info("Читаю файлы 01–14 из «%s».", data_dir)
    raw = load_all(data_dir, settings, cache_dir)
    logger.info("Строю таблицы.")
    return build_from_raw(raw, settings)


def build_from_raw(raw: dict[str, pd.DataFrame], settings: dict) -> dict[str, pd.DataFrame]:
    """ls_info, panel и measure_events из таблиц load_all."""
    info, osv, payments, events = raw["info"], raw["osv"], raw["payments"], raw["events"]
    panel = build_panel(osv, payments, settings)
    return {
        "ls_info": build_ls_info(info, osv["ls"]),
        "panel": panel,
        "measure_events": build_measure_events(events, panel, payments, settings),
    }


def save_tables(tables: dict[str, pd.DataFrame], output_dir: Path | str, plan_month: pd.Timestamp) -> Path:
    """Пишет таблицы в output/<YYYY-MM>/<имя>.parquet и возвращает папку."""
    target = Path(output_dir) / f"{plan_month:%Y-%m}"
    target.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(target / f"{name}.parquet", index=False)
    return target


def parse_plan_month(text: str) -> pd.Timestamp:
    try:
        return pd.to_datetime(text, format="%Y-%m")
    except ValueError as error:
        raise ValueError(f"Неверный месяц «{text}»: нужен формат ГГГГ-ММ, например 2026-03.") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Строит таблицы ls_info, panel и measure_events.")
    parser.add_argument("--config", help="файл настроек (по умолчанию configs/settings.yaml)")
    parser.add_argument("--data", help="папка с файлами 01–14 (по умолчанию из настроек)")
    parser.add_argument("--out", help="папка для результатов (по умолчанию из настроек)")
    parser.add_argument("--month", help="месяц плана ГГГГ-ММ (по умолчанию последний месяц панели)")
    parser.add_argument("--no-cache", action="store_true", help="читать исходные файлы, не используя кэш")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        settings = load_settings(args.config)
        plan_month = parse_plan_month(args.month) if args.month else None
        cache_dir = None if args.no_cache else default_cache_dir(settings)
        tables = build_tables(args.data or REPO_ROOT / settings["paths"]["data_dir"], settings, cache_dir)
        first, last = tables["panel"]["month"].min(), tables["panel"]["month"].max()
        plan_month = plan_month if plan_month is not None else last
        if not first <= plan_month <= last:
            logger.warning(
                "В панели нет строк за месяц плана %s: данные покрывают %s…%s.",
                f"{plan_month:%Y-%m}", f"{first:%Y-%m}", f"{last:%Y-%m}",
            )
        target = save_tables(tables, args.out or REPO_ROOT / settings["paths"]["output_dir"], plan_month)
    except (OSError, ValueError, KeyError) as error:
        logger.error("Таблицы не построены: %s", error)
        return 1

    for name, table in tables.items():
        logger.info("%s: %d строк, %d ЛС.", name, len(table), table["ls"].nunique())
    logger.info("Результат записан в «%s».", target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
