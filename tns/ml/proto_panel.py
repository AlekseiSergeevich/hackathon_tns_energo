"""Временная прото-панель для ML.

Строит таблицы ls_info, panel и measure_events в формате docs/interfaces.md
прямо из файлов 01–13. Нужна, пока нет настоящей панели из tns/data (DATA-4).
Когда она появится, ML переключится на неё, а этот модуль будет удалён.

Отличия от будущей панели:
- dz_overdue и overdue_m — упрощённый FIFO по сальдо (подсказка к DATA-5 в плане);
- в measure_events только колонки, нужные для признаков (без результата меры).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

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
MEASURES = list(MEASURE_FILES)

# У этих мер в файлах всегда 1-е число: известен только месяц.
MONTH_PRECISION = {"autodial", "email", "sms", "claim", "visit"}

# TODO(ORG-1): читать из configs/settings.yaml (result.share, norm.proxy_months).
RESULT_SHARE = 0.30
NORM_MAX_MONTHS = 12
NORM_MIN_MONTHS = 3

# Сколько прошлых начислений максимум просматриваем при расчёте срока просрочки.
OVERDUE_MAX_LAG = 24

OSV_METRICS = {"сз на начало": "saldo_start", "начислено": "accrual", "опалачено": "paid", "оплачено": "paid"}
HOUSE_TYPES = ["ЧД", "МКД", "Общежитие"]


# ---------------------------------------------------------------- загрузка файлов

def _yes_no(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    return text.map({"да": True, "нет": False}).astype("boolean")


def load_ls_info(data_dir: Path) -> pd.DataFrame:
    """Файл 01 → ls_info."""
    df = pd.read_excel(data_dir / "01 Общая информация о ЛС ХК.xlsx")
    df.columns = [str(c).strip() for c in df.columns]

    house = pd.Series(pd.NA, index=df.index, dtype="object")
    for name in reversed(HOUSE_TYPES):  # при нескольких «Да» побеждает первый по списку
        house = house.mask(_yes_no(df[name]).fillna(False).astype(bool), name)

    out = pd.DataFrame(
        {
            "ls": pd.to_numeric(df["ЛС"], errors="coerce"),
            "has_phone": _yes_no(df["Наличие телефона"]),
            "has_ebill": _yes_no(df["электронная квитанция"]),
            "has_benefits": _yes_no(df["Наличие льгот"]),
            "not_residing": _yes_no(df["не проживает"]),
            "remote_disconnect": _yes_no(df["Возможность дистанционного отключения"]),
            "house_type": pd.Categorical(house, categories=HOUSE_TYPES),
            "gasified": _yes_no(df["Газификация дома"]),
            "address_guid": df["Адрес (ГУИД)"].astype("string"),
        }
    )
    out = out.dropna(subset=["ls"]).drop_duplicates("ls")
    out["ls"] = out["ls"].astype("int64")
    return out.reset_index(drop=True)


def load_osv(data_dir: Path) -> pd.DataFrame:
    """Файл 02 → длинная таблица ls, month, saldo_start, accrual, paid.

    Повтор месяца в шапке считается следующим месяцем (osv.duplicate_month: shift_next).
    """
    path = data_dir / "02 Обортно-сальдовая ведомость ЛС ХК.xlsx"
    raw = pd.read_excel(path, header=[0, 1])
    ls = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    cols = list(raw.columns[1:])
    if len(cols) % 3:
        raise ValueError(f"ОСВ: ожидались блоки по 3 колонки на месяц, а колонок {len(cols)}.")

    frames, seen = [], []
    for i in range(0, len(cols), 3):
        block = cols[i : i + 3]
        month = pd.Timestamp(block[0][0]).to_period("M").to_timestamp()
        metrics = [OSV_METRICS.get(str(c[1]).strip().lower().split(".")[0]) for c in block]
        if metrics != ["saldo_start", "accrual", "paid"]:
            raise ValueError(f"ОСВ: непонятные показатели в блоке {block[0][0]}: {[c[1] for c in block]}.")
        if month in seen:
            shifted = month + pd.DateOffset(months=1)
            if month != seen[-1] or shifted in seen:
                raise ValueError(f"ОСВ: месяц {month:%Y-%m} повторяется не подряд, разобрать шапку нельзя.")
            log.warning("ОСВ: месяц %s повторяется в колонках %d–%d, считаем его %s.",
                        f"{month:%Y-%m}", i + 2, i + 4, f"{shifted:%Y-%m}")
            month = shifted
        seen.append(month)
        frames.append(
            pd.DataFrame(
                {
                    "ls": ls,
                    "month": month,
                    "saldo_start": pd.to_numeric(raw[block[0]], errors="coerce"),
                    "accrual": pd.to_numeric(raw[block[1]], errors="coerce"),
                    "paid": pd.to_numeric(raw[block[2]], errors="coerce"),
                }
            )
        )

    expected = pd.date_range(seen[0], seen[-1], freq="MS")
    if list(expected) != seen:
        raise ValueError("ОСВ: месяцы в шапке идут не подряд.")

    osv = pd.concat(frames, ignore_index=True).dropna(subset=["ls"])
    osv["ls"] = osv["ls"].astype("int64")
    return osv.drop_duplicates(["ls", "month"]).sort_values(["ls", "month"]).reset_index(drop=True)


def load_payments(data_dir: Path) -> pd.DataFrame:
    """Файл 03 → ls, payment_date, amount."""
    df = pd.read_csv(data_dir / "03 Оплаты ХК.csv", sep=";", encoding="utf-8-sig", dtype="string")
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame(
        {
            "ls": pd.to_numeric(df["Номер"], errors="coerce"),
            "payment_date": pd.to_datetime(df["Дата оплаты"], dayfirst=True, errors="coerce"),
            "amount": pd.to_numeric(
                df["Сумма"].str.replace(" ", "").str.replace(" ", "").str.replace(",", "."),
                errors="coerce",
            ),
        }
    )
    lost = out.isna().any(axis=1).sum()
    if lost:
        log.warning("Оплаты: %d строк не разобрано (пустой ЛС, дата или сумма), они пропущены.", lost)
    out = out.dropna()
    out["ls"] = out["ls"].astype("int64")
    out["amount"] = out["amount"].astype("float64")
    return out.reset_index(drop=True)


def _read_measure_file(path: Path, measure: str) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    first = raw.iloc[:, 0].astype("string").str.strip().str.lower()
    header_rows = first.index[first.isin(["лс", "номер"])]
    if len(header_rows) == 0:
        raise ValueError(f"В файле «{path.name}» не найдена строка заголовка «ЛС | Дата».")
    data = raw.iloc[header_rows[0] + 1 :, :2]
    out = pd.DataFrame(
        {
            "ls": pd.to_numeric(data.iloc[:, 0], errors="coerce"),
            "event_date": pd.to_datetime(data.iloc[:, 1], errors="coerce"),
        }
    )
    lost = out.isna().any(axis=1)
    if lost.any():
        log.warning("«%s»: %d из %d строк без ЛС или даты (например, #N/A в выгрузке), они пропущены.",
                    path.name, lost.sum(), len(out))
    out = out[~lost]
    out["ls"] = out["ls"].astype("int64")
    out["measure"] = measure
    return out


def load_measure_events(data_dir: Path) -> pd.DataFrame:
    """Файлы 04–13 → журнал мер, одна строка = ЛС × мера × месяц."""
    raw = pd.concat(
        [_read_measure_file(data_dir / f, m) for m, f in MEASURE_FILES.items()], ignore_index=True
    )
    raw["month"] = raw["event_date"].dt.to_period("M").dt.to_timestamp()
    events = (
        raw.groupby(["ls", "measure", "month"], as_index=False)
        .agg(event_date=("event_date", "min"), n_raw=("event_date", "size"))
    )
    events["measure"] = pd.Categorical(events["measure"], categories=MEASURES)
    events["date_precision"] = pd.Categorical(
        np.where(events["measure"].isin(MONTH_PRECISION), "month", "day"), categories=["month", "day"]
    )
    collapsed = len(raw) - len(events)
    if collapsed:
        log.info("Журнал мер: %d повторных строк в том же месяце свёрнуто.", collapsed)
    return events[["ls", "measure", "month", "event_date", "date_precision", "n_raw"]]


# ---------------------------------------------------------------- панель

def _overdue(panel: pd.DataFrame) -> pd.DataFrame:
    """dz_overdue, overdue_m, overdue_censored по FIFO на сальдо.

    При FIFO непогашенный остаток — самые свежие начисления. Начисление за M−1
    на 01.M ещё не просрочено, остальное просрочено. Срок = сколько начислений
    M−2, M−3, … нужно сложить, чтобы покрыть просроченную часть.
    """
    g = panel.groupby("ls", sort=False)["accrual"]
    # Отрицательное начисление (перерасчёт) — свежего счёта нет, весь долг просрочен.
    fresh_bill = g.shift(1).clip(lower=0)
    overdue = (panel["dz"] - fresh_bill).clip(lower=0)  # NaN в первом месяце данных

    covered = pd.Series(0.0, index=panel.index)
    months = pd.Series(np.nan, index=panel.index)
    available = pd.Series(0, index=panel.index)
    need = overdue > 0
    for lag in range(2, OVERDUE_MAX_LAG + 2):
        acc = g.shift(lag)
        has = acc.notna()
        available += has.astype(int)
        covered += acc.fillna(0)
        hit = need & has & months.isna() & (covered >= overdue)
        months[hit] = lag - 1

    censored = need & months.isna()
    months[censored] = available[censored]  # долг старше начала данных: срок занижен
    months[overdue == 0] = 0
    return pd.DataFrame(
        {
            "dz_overdue": overdue,
            "overdue_m": months.round().astype("Int64"),
            "overdue_censored": censored.astype("boolean").mask(overdue.isna()),
        }
    )


def _months_no_payment(panel: pd.DataFrame) -> pd.DataFrame:
    """Сколько месяцев подряд перед M не было оплат (по файлу 03)."""
    has_pay = panel["paid_payments"].fillna(0) > 0
    block = has_pay.groupby(panel["ls"], sort=False).cumsum()
    run = (~has_pay).astype(int).groupby([panel["ls"], block], sort=False).cumsum()
    g_run = run.groupby(panel["ls"], sort=False).shift(1)
    g_block = block.groupby(panel["ls"], sort=False).shift(1)
    return pd.DataFrame(
        {
            "months_no_payment": g_run.astype("Int64"),
            "months_no_payment_censored": ((g_block == 0) & (g_run > 0)).astype("boolean").mask(g_run.isna()),
        }
    )


def _norm_proxy(panel: pd.DataFrame) -> pd.Series:
    """Среднее начисление за доступные месяцы ≤ M−1 (до 12, минимум 3)."""
    acc, ls = panel["accrual"], panel["ls"]
    csum = acc.fillna(0).groupby(ls, sort=False).cumsum()
    ccnt = acc.notna().astype(int).groupby(ls, sort=False).cumsum()
    prev_sum = csum.groupby(ls, sort=False).shift(1)
    prev_cnt = ccnt.groupby(ls, sort=False).shift(1)
    old_sum = csum.groupby(ls, sort=False).shift(NORM_MAX_MONTHS + 1).fillna(0)
    old_cnt = ccnt.groupby(ls, sort=False).shift(NORM_MAX_MONTHS + 1).fillna(0)
    n = prev_cnt - old_cnt
    mean = (prev_sum - old_sum) / n
    return mean.where(n >= NORM_MIN_MONTHS)


def build_panel(osv: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    """ОСВ + оплаты → panel в формате interfaces.md."""
    panel = osv.sort_values(["ls", "month"]).reset_index(drop=True)
    pay_m = (
        payments.assign(month=payments["payment_date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["ls", "month"], as_index=False)["amount"].sum()
        .rename(columns={"amount": "paid_payments"})
    )
    panel = panel.merge(pay_m, on=["ls", "month"], how="left")
    panel["paid_payments"] = panel["paid_payments"].fillna(0.0)

    panel["dz"] = panel["saldo_start"].clip(lower=0)
    panel = pd.concat([panel, _overdue(panel), _months_no_payment(panel)], axis=1)
    panel["norm"] = _norm_proxy(panel)
    panel["norm_source"] = pd.Categorical(
        np.where(panel["norm"].notna(), "proxy", None), categories=["customer", "proxy"]
    )
    panel["pay30_m"] = (panel["paid_payments"] >= RESULT_SHARE * panel["dz"]).astype("boolean")
    panel.loc[~(panel["dz"] > 0), "pay30_m"] = pd.NA

    cols = [
        "ls", "month", "saldo_start", "dz", "dz_overdue", "overdue_m", "overdue_censored",
        "months_no_payment", "months_no_payment_censored", "norm", "norm_source",
        "accrual", "paid", "paid_payments", "pay30_m",
    ]
    return panel[cols]


# ---------------------------------------------------------------- всё вместе с кэшем

def load_proto_tables(data_dir: Path, cache_dir: Path, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """ls_info, panel, measure_events (+ payments для сверок). Кэш в parquet.

    Кэш пересобирается, если исходный файл или этот модуль новее кэша, или refresh=True.
    """
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Папка с данными не найдена: {data_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    names = ["ls_info", "panel", "measure_events", "payments"]
    paths = {n: cache_dir / f"{n}.parquet" for n in names}

    sources = [p for p in data_dir.iterdir() if p.suffix in {".xlsx", ".csv"}] + [Path(__file__)]
    newest_src = max(p.stat().st_mtime for p in sources)
    fresh = all(p.exists() and p.stat().st_mtime > newest_src for p in paths.values())
    if fresh and not refresh:
        log.info("Прото-панель: читаю кэш из %s", cache_dir)
        return {n: pd.read_parquet(p) for n, p in paths.items()}

    log.info("Прото-панель: читаю исходные файлы (несколько минут)…")
    payments = load_payments(data_dir)
    tables = {
        "ls_info": load_ls_info(data_dir),
        "panel": build_panel(load_osv(data_dir), payments),
        "measure_events": load_measure_events(data_dir),
        "payments": payments,
    }
    for n, df in tables.items():
        df.to_parquet(paths[n], index=False)
    log.info("Прото-панель: кэш сохранён в %s", cache_dir)
    return tables
