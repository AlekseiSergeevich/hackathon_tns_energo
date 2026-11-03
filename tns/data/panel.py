"""Таблицы данных для остальных частей: ls_info, panel, measure_events.

Форматы и смысл колонок — в docs/interfaces.md (разделы 1–3).

Запуск:
    python -m tns.data.panel [--data ПАПКА] [--out ПАПКА] [--month YYYY-MM] [--config ФАЙЛ]

Колонки panel с known_at = start считаются только по сальдо на 01.M и по данным месяцев
до M. Колонки end (accrual, paid, paid_payments, pay30_m) содержат будущее относительно
среза M и в признаки месяца M попадать не должны.

Чтение файлов (функции read_*) временно живёт здесь и переедет в tns/data/loaders.py (DATA-1).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS = REPO_ROOT / "configs" / "settings.yaml"

INFO_FILE = "01 Общая информация о ЛС ХК.xlsx"
OSV_FILE = "02 Обортно-сальдовая ведомость ЛС ХК.xlsx"
PAYMENTS_FILE = "03 Оплаты ХК.csv"
MEASURE_FILES = {
    "autodial": "04 Автодозвон ХК.xlsx",
    "email": "05 E-mail ХК.xlsx",
    "sms": "06 СМС ХК.xlsx",
    "operator_call": "07 Обзвон оператором ХК.xlsx",
    "claim": "08 Претензия ХК.xlsx",
    "visit": "09 Выезд к абоненту ХК.xlsx",
    "restriction_notice": "10 Уведомление о введении ограничения ХК.xlsx",
    "restriction": "11 Ограничение ХК.xlsx",
    "court_application": "12 Заявление о выдаче судебного приказа ХК.xlsx",
    "court_result": "13 Получение судебного приказа или ИЛ ХК.xlsx",
}
# Меры, у которых в данных известен только месяц: дата проставлена 1-м числом.
MONTH_PRECISION_MEASURES = {"autodial", "email", "sms", "claim", "visit"}

OSV_METRICS = {
    "сз на начало": "saldo_start",
    "начислено": "accrual",
    "оплачено": "paid",
    "опалачено": "paid",  # опечатка в шапке файла 02
}
INFO_FLAGS = {
    "has_phone": "Наличие телефона",
    "has_ebill": "электронная квитанция",
    "has_benefits": "Наличие льгот",
    "not_residing": "не проживает",
    "remote_disconnect": "Возможность дистанционного отключения",
    "gasified": "Газификация дома",
}
HOUSE_TYPES = ["ЧД", "МКД", "Общежитие"]
INFO_GUID_COLUMN = "Адрес (ГУИД)"
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


def load_settings(path: Path | str | None = None) -> dict:
    """Читает configs/settings.yaml."""
    path = Path(path) if path else DEFAULT_SETTINGS
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл настроек: {path}")
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Чтение исходных файлов
# ---------------------------------------------------------------------------

def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Не найден входной файл: {path}")
    return path


def _require_columns(df: pd.DataFrame, columns: list[str], file_name: str) -> None:
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"В файле «{file_name}» нет колонки «{column}».")


def _clean_ls(df: pd.DataFrame, column: str, file_name: str) -> pd.DataFrame:
    """Приводит номер ЛС к int64. Строки без корректного номера убирает и пишет об этом в лог."""
    ls = pd.to_numeric(df[column], errors="coerce").astype("float64")
    valid = ls.notna() & (ls % 1 == 0)
    if not valid.all():
        logger.warning("Файл «%s»: убрано %d строк без номера ЛС.", file_name, int((~valid).sum()))
    df = df.loc[valid].copy()
    df["ls"] = ls[valid].astype("int64")
    return df


def _month_start(dates: pd.Series) -> pd.Series:
    return dates.dt.to_period("M").dt.to_timestamp().astype("datetime64[ns]")


def read_info(data_dir: Path) -> pd.DataFrame:
    """Файл 01: анкета ЛС. Колонки с флагами остаются строками «Да» / «Нет»."""
    raw = pd.read_excel(_require_file(data_dir / INFO_FILE))
    raw.columns = [str(c).strip() for c in raw.columns]
    columns = [*INFO_FLAGS.values(), *HOUSE_TYPES, INFO_GUID_COLUMN]
    _require_columns(raw, ["ЛС", *columns], INFO_FILE)

    df = _clean_ls(raw, "ЛС", INFO_FILE)[["ls", *columns]]
    duplicated = df.duplicated("ls")
    if duplicated.any():
        conflicts = df.groupby("ls")[columns].nunique(dropna=False).gt(1).any(axis=1).sum()
        logger.warning(
            "Файл «%s»: %d повторов ЛС, оставлена первая строка (нужные поля различаются у %d ЛС).",
            INFO_FILE, int(duplicated.sum()), int(conflicts),
        )
    return df.loc[~duplicated].reset_index(drop=True)


def _parse_osv_blocks(header: pd.DataFrame) -> tuple[int, list[tuple[pd.Timestamp, dict]], list[int]]:
    """Разбирает двухуровневую шапку ОСВ.

    Возвращает номер колонки ЛС, блоки (месяц, {показатель: номер колонки}) и индексы блоков,
    которые в шапке повторяют уже встреченный месяц.
    """
    top, sub = header.iloc[0].tolist(), header.iloc[1].tolist()
    ls_columns = [i for i, value in enumerate(top) if str(value).strip() == "ЛС"]
    if not ls_columns:
        raise ValueError(f"В файле «{OSV_FILE}» нет колонки «ЛС».")

    blocks: list[tuple[pd.Timestamp, dict]] = []
    for i, value in enumerate(top):
        if i == ls_columns[0]:
            continue
        if pd.notna(value):
            month = pd.to_datetime(value, errors="coerce")
            if pd.isna(month):
                raise ValueError(
                    f"В файле «{OSV_FILE}», колонка {i + 1}: «{value}» не похоже на месяц."
                )
            blocks.append((month.to_period("M").to_timestamp(), {}))
        metric = OSV_METRICS.get(str(sub[i]).strip().lower())
        if metric and blocks:
            blocks[-1][1].setdefault(metric, i)

    if not blocks:
        raise ValueError(f"В файле «{OSV_FILE}» в шапке не найдено ни одного месяца.")
    for month, metrics in blocks:
        missing = {"saldo_start", "accrual", "paid"} - set(metrics)
        if missing:
            raise ValueError(
                f"В файле «{OSV_FILE}» для месяца {month:%Y-%m} нет показателей: "
                f"{', '.join(sorted(missing))}."
            )

    seen: set[pd.Timestamp] = set()
    repeated = []
    for index, (month, _) in enumerate(blocks):
        if month in seen:
            repeated.append(index)
        seen.add(month)
    return ls_columns[0], blocks, repeated


def read_osv(data_dir: Path, settings: dict) -> pd.DataFrame:
    """Файл 02 в длинном формате: ls, month, saldo_start, accrual, paid."""
    path = _require_file(data_dir / OSV_FILE)
    policy = settings["osv"]["duplicate_month"]
    if policy not in {"shift_next", "drop", "error"}:
        raise ValueError(f"Неизвестное значение osv.duplicate_month: «{policy}».")
    header = pd.read_excel(path, header=None, nrows=2)
    ls_column, blocks, repeated = _parse_osv_blocks(header)

    body = pd.read_excel(path, header=None, skiprows=2)
    body = _clean_ls(body, ls_column, OSV_FILE)
    duplicated = body.duplicated("ls")
    if duplicated.any():
        logger.warning("Файл «%s»: %d повторов ЛС, оставлена первая строка.", OSV_FILE, int(duplicated.sum()))
        body = body.loc[~duplicated]

    known_months = {month for month, _ in blocks}
    drop_blocks = set()
    for index in repeated:
        month, metrics = blocks[index]
        previous = next(m for mo, m in blocks[:index] if mo == month)
        expected = body[previous["saldo_start"]] + body[previous["accrual"]] - body[previous["paid"]]
        actual = body[metrics["saldo_start"]]
        nonzero = (actual != 0) | (expected != 0)
        share = (expected - actual).abs().lt(0.01)[nonzero].mean() if nonzero.any() else float("nan")
        logger.warning(
            "Файл «%s»: месяц %s повторяется в шапке (колонки с %d). У %.1f %% ненулевых строк "
            "сальдо блока продолжает предыдущий блок. Настройка osv.duplicate_month: %s.",
            OSV_FILE, f"{month:%Y-%m}", min(metrics.values()) + 1, share * 100, policy,
        )
        if policy == "error":
            raise ValueError(f"В файле «{OSV_FILE}» месяц {month:%Y-%m} повторяется в шапке.")
        if policy == "drop":
            drop_blocks.add(index)
            continue
        next_month = month + pd.DateOffset(months=1)
        if next_month in known_months:
            raise ValueError(
                f"В файле «{OSV_FILE}» месяц {month:%Y-%m} повторяется, а следующий месяц "
                f"{next_month:%Y-%m} уже есть в шапке: сдвинуть блок нельзя."
            )
        known_months.add(next_month)
        blocks[index] = (next_month, metrics)

    frames = []
    for index, (month, metrics) in enumerate(blocks):
        if index in drop_blocks:
            continue
        frame = pd.DataFrame({"ls": body["ls"], "month": month})
        for metric, column in metrics.items():
            frame[metric] = pd.to_numeric(body[column], errors="coerce").astype("float64")
        frames.append(frame)
    osv = pd.concat(frames, ignore_index=True)
    osv["month"] = osv["month"].astype("datetime64[ns]")

    lost = osv[["saldo_start", "accrual", "paid"]].isna().sum().sum()
    if lost:
        logger.warning("Файл «%s»: %d значений не удалось прочитать как число.", OSV_FILE, int(lost))
    months = pd.date_range(osv["month"].min(), osv["month"].max(), freq="MS")
    if len(months) != osv["month"].nunique():
        raise ValueError(f"В файле «{OSV_FILE}» в шапке пропущены месяцы между {months[0]:%Y-%m} и {months[-1]:%Y-%m}.")
    return osv


def read_payments(data_dir: Path) -> pd.DataFrame:
    """Файл 03: ls, payment_date, amount.

    Одинаковые строки не удаляем: ОСВ сходится с суммой вместе с ними, то есть это
    отдельные платежи. Отрицательные суммы (сторно) тоже остаются.
    """
    raw = pd.read_csv(_require_file(data_dir / PAYMENTS_FILE), sep=";", encoding="utf-8-sig", dtype=str)
    raw.columns = [str(c).strip() for c in raw.columns]
    _require_columns(raw, ["Номер", "Дата оплаты", "Сумма"], PAYMENTS_FILE)

    df = _clean_ls(raw, "Номер", PAYMENTS_FILE)
    df["payment_date"] = pd.to_datetime(df["Дата оплаты"], format="%d.%m.%Y", errors="coerce")
    amount = df["Сумма"].str.replace(r"[\s ]", "", regex=True).str.replace(",", ".", regex=False)
    df["amount"] = pd.to_numeric(amount, errors="coerce").astype("float64")
    invalid = df["payment_date"].isna() | df["amount"].isna()
    if invalid.any():
        logger.warning("Файл «%s»: убрано %d строк с неверной датой или суммой.", PAYMENTS_FILE, int(invalid.sum()))
    df = df.loc[~invalid, ["ls", "payment_date", "amount"]].reset_index(drop=True)
    df["payment_date"] = df["payment_date"].astype("datetime64[ns]")
    return df


def read_measure_events(data_dir: Path) -> pd.DataFrame:
    """Файлы 04–13: ls, measure, event_date. Первая строка файла — название меры, вторая — шапка."""
    frames = []
    for measure, file_name in MEASURE_FILES.items():
        raw = pd.read_excel(_require_file(data_dir / file_name), header=1)
        raw.columns = [str(c).strip() for c in raw.columns]
        _require_columns(raw, ["ЛС", "Дата"], file_name)
        df = _clean_ls(raw, "ЛС", file_name)
        df["event_date"] = pd.to_datetime(df["Дата"], errors="coerce")
        invalid = df["event_date"].isna()
        if invalid.any():
            logger.warning("Файл «%s»: убрано %d строк с неверной датой.", file_name, int(invalid.sum()))
        df = df.loc[~invalid, ["ls", "event_date"]]
        df["measure"] = measure
        frames.append(df)
    events = pd.concat(frames, ignore_index=True)
    events["event_date"] = events["event_date"].astype("datetime64[ns]")
    return events


# ---------------------------------------------------------------------------
# ls_info
# ---------------------------------------------------------------------------

def _yes_no_to_boolean(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.lower()
    return text.map({"да": True, "нет": False}, na_action="ignore").astype("boolean")


def build_ls_info(info: pd.DataFrame, osv_ls: pd.Series) -> pd.DataFrame:
    """Анкета на все ЛС из файлов 01 и 02. ЛС только из 02 получают пустые флаги."""
    result = pd.DataFrame({"ls": info["ls"].astype("int64")})
    for column, source in INFO_FLAGS.items():
        result[column] = _yes_no_to_boolean(info[source]).array

    # Тип дома: первый отмеченный флаг. Если ни один не отмечен — неизвестно.
    is_set = np.column_stack([_yes_no_to_boolean(info[h]).fillna(False).to_numpy(bool) for h in HOUSE_TYPES])
    if (is_set.sum(axis=1) > 1).any():
        logger.warning("Файл «%s»: у %d ЛС отмечено несколько типов дома, взят первый.",
                       INFO_FILE, int((is_set.sum(axis=1) > 1).sum()))
    codes = np.where(is_set.any(axis=1), is_set.argmax(axis=1), -1)
    result["house_type"] = pd.Categorical.from_codes(codes, categories=HOUSE_TYPES)
    result["address_guid"] = info[INFO_GUID_COLUMN].astype("string").array

    only_in_osv = np.setdiff1d(osv_ls.unique(), result["ls"].to_numpy())
    only_in_info = np.setdiff1d(result["ls"].to_numpy(), osv_ls.unique())
    if len(only_in_osv) or len(only_in_info):
        logger.warning("ЛС только в файле 02: %d, только в файле 01: %d.", len(only_in_osv), len(only_in_info))
    if len(only_in_osv):
        missing = pd.DataFrame({"ls": only_in_osv.astype("int64")})
        result = pd.concat([result, missing], ignore_index=True)
        for column in INFO_FLAGS:
            result[column] = result[column].astype("boolean")
        result["address_guid"] = result["address_guid"].astype("string")
        result["house_type"] = pd.Categorical(result["house_type"], categories=HOUSE_TYPES)
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

    grouped = events.assign(month=_month_start(events["event_date"])).groupby(["ls", "measure", "month"], as_index=False)
    log = grouped.agg(event_date=("event_date", "min"), n_raw=("event_date", "size"))
    collapsed = log.loc[log["n_raw"] > 1].groupby("measure")["n_raw"].apply(lambda n: int((n - 1).sum()))
    if len(collapsed):
        logger.info("Журнал мер: свёрнуто повторов в месяце: %s.", collapsed.to_dict())

    log["n_raw"] = log["n_raw"].astype("int64")
    log["date_precision"] = pd.Categorical(
        np.where(log["measure"].isin(MONTH_PRECISION_MEASURES), "month", "day"), categories=["month", "day"]
    )
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

def build_tables(data_dir: Path | str, settings: dict) -> dict[str, pd.DataFrame]:
    """Читает файлы 01–13 и строит ls_info, panel и measure_events."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Не найдена папка с данными: {data_dir}")

    logger.info("Читаю файлы 01–13 из «%s».", data_dir)
    info = read_info(data_dir)
    osv = read_osv(data_dir, settings)
    payments = read_payments(data_dir)
    events = read_measure_events(data_dir)

    logger.info("Строю таблицы.")
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


def _parse_plan_month(text: str) -> pd.Timestamp:
    try:
        return pd.to_datetime(text, format="%Y-%m")
    except ValueError as error:
        raise ValueError(f"Неверный месяц «{text}»: нужен формат ГГГГ-ММ, например 2026-03.") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Строит таблицы ls_info, panel и measure_events.")
    parser.add_argument("--config", help="файл настроек (по умолчанию configs/settings.yaml)")
    parser.add_argument("--data", help="папка с файлами 01–13 (по умолчанию из настроек)")
    parser.add_argument("--out", help="папка для результатов (по умолчанию из настроек)")
    parser.add_argument("--month", help="месяц плана ГГГГ-ММ (по умолчанию последний месяц панели)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        settings = load_settings(args.config)
        plan_month = _parse_plan_month(args.month) if args.month else None
        tables = build_tables(args.data or REPO_ROOT / settings["paths"]["data_dir"], settings)
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
