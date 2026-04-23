#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


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

INFORMATIONAL_MEASURES = {"sms", "autodial", "email"}
HUMAN_INTENSIVE_MEASURES = {"operator_call", "visit"}


@dataclass
class BuildContext:
    data_dir: Path
    snapshot_month: pd.Timestamp
    asof_date: pd.Timestamp
    window12_start: pd.Timestamp
    window6_start: pd.Timestamp
    window3_start: pd.Timestamp


def normalize_ls(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def yes_no_to_label(series: pd.Series) -> pd.Series:
    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .replace(
            {
                "да": "Yes",
                "нет": "No",
                "true": "Yes",
                "false": "No",
                "1": "Yes",
                "0": "No",
            }
        )
    )
    return normalized.fillna("Unknown")


def parse_money(series: pd.Series) -> pd.Series:
    text = (
        series.astype("string")
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(text, errors="coerce")


def ensure_column(df: pd.DataFrame, col_name: str, default: str = "Unknown") -> pd.Series:
    if col_name in df.columns:
        return df[col_name]
    return pd.Series([default] * len(df), index=df.index, dtype="string")


def to_month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)


def linear_slope(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    y = y[~np.isnan(y)]
    if y.size < 2:
        return np.nan
    x = np.arange(y.size, dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def max_consecutive_nonpayment(values: Iterable[float]) -> int:
    max_run = 0
    run = 0
    for v in values:
        if pd.isna(v) or float(v) <= 0:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return int(max_run)


def load_general_info(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "01 Общая информация о ЛС ХК.xlsx"
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    df["ls"] = normalize_ls(df["ЛС"])

    room_type = np.select(
        [
            yes_no_to_label(ensure_column(df, "ЧД")) == "Yes",
            yes_no_to_label(ensure_column(df, "МКД")) == "Yes",
            yes_no_to_label(ensure_column(df, "Общежитие")) == "Yes",
        ],
        ["ЧД", "МКД", "Общежитие"],
        default="Unknown",
    )
    meter_installation = np.select(
        [
            yes_no_to_label(ensure_column(df, "Установка Тамбур")) == "Yes",
            yes_no_to_label(ensure_column(df, "Установка опора")) == "Yes",
            yes_no_to_label(ensure_column(df, "Установка в квартире/доме")) == "Yes",
            yes_no_to_label(ensure_column(df, "Установка лестничкая клетка")) == "Yes",
        ],
        ["Тамбур", "Опора", "В квартире/доме", "Лестничная клетка"],
        default="Unknown",
    )
    settlement_type = np.where(
        yes_no_to_label(ensure_column(df, "Город")) == "Yes", "Город", "Село/Прочее"
    )

    result = pd.DataFrame(
        {
            "ls": df["ls"],
            "has_phone": yes_no_to_label(ensure_column(df, "Наличие телефона")),
            "has_ebill": yes_no_to_label(ensure_column(df, "электронная квитанция")),
            "is_gasified": yes_no_to_label(ensure_column(df, "Газификация дома")),
            "has_benefits": yes_no_to_label(ensure_column(df, "Наличие льгот")),
            "not_residing": yes_no_to_label(ensure_column(df, "не проживает")),
            "remote_disconnect_possible": yes_no_to_label(
                ensure_column(df, "Возможность дистанционного отключения")
            ),
            "room_type": pd.Series(room_type, dtype="string"),
            "meter_installation": pd.Series(meter_installation, dtype="string"),
            "settlement_type": pd.Series(settlement_type, dtype="string"),
        }
    )
    return result.dropna(subset=["ls"]).drop_duplicates("ls")


def load_osv_long(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "02 Обортно-сальдовая ведомость ЛС ХК.xlsx"
    raw = pd.read_excel(path, header=[0, 1])
    ls_candidates = [col for col in raw.columns if str(col[0]).strip() == "ЛС"]
    if not ls_candidates:
        raise ValueError("Не удалось найти колонку ЛС в ОСВ-файле.")
    raw["ls"] = normalize_ls(raw[ls_candidates[0]])

    metric_alias = {
        "сз на начало": "sz_start",
        "сз на начало.1": "sz_start",
        "начислено": "accrual",
        "начислено.1": "accrual",
        "опалачено": "paid",
        "оплачено": "paid",
        "опалачено.1": "paid",
        "оплачено.1": "paid",
    }

    records: list[pd.DataFrame] = []
    for month in raw.columns.get_level_values(0).unique():
        month_ts = pd.to_datetime(month, errors="coerce")
        if pd.isna(month_ts):
            continue

        month_columns = raw.loc[:, raw.columns.get_level_values(0) == month]
        selected = {}
        for _, metric_name in month_columns.columns:
            normalized_metric = metric_alias.get(str(metric_name).strip().lower())
            if normalized_metric and normalized_metric not in selected:
                selected[normalized_metric] = (month, metric_name)

        if {"sz_start", "accrual", "paid"} - set(selected):
            continue

        frame = pd.DataFrame(
            {
                "ls": raw["ls"],
                "month": to_month_start(pd.Timestamp(month_ts)),
                "sz_start": pd.to_numeric(raw[selected["sz_start"]], errors="coerce"),
                "accrual": pd.to_numeric(raw[selected["accrual"]], errors="coerce"),
                "paid": pd.to_numeric(raw[selected["paid"]], errors="coerce"),
            }
        )
        records.append(frame)

    long_df = pd.concat(records, ignore_index=True)
    long_df = long_df.dropna(subset=["ls"]).sort_values(["ls", "month"])
    return long_df


def compute_osv_features(osv_long: pd.DataFrame, ctx: BuildContext) -> pd.DataFrame:
    current = (
        osv_long[osv_long["month"] == ctx.snapshot_month][["ls", "sz_start"]]
        .rename(columns={"sz_start": "DZ_current"})
        .drop_duplicates("ls")
    )

    win12 = osv_long[
        (osv_long["month"] >= ctx.window12_start) & (osv_long["month"] <= ctx.snapshot_month)
    ]
    win6 = osv_long[
        (osv_long["month"] >= ctx.window6_start) & (osv_long["month"] <= ctx.snapshot_month)
    ]
    win3 = osv_long[
        (osv_long["month"] >= ctx.window3_start) & (osv_long["month"] <= ctx.snapshot_month)
    ]

    agg12 = (
        win12.groupby("ls", dropna=False)
        .agg(
            avg_accrual_12m=("accrual", "mean"),
            avg_payment_12m=("paid", "mean"),
            accrual_sum_12m=("accrual", "sum"),
            paid_sum_12m=("paid", "sum"),
            months_with_full_payment_12m=("paid", lambda s: int((s >= win12.loc[s.index, "accrual"]).sum())),
            months_with_zero_payment_12m=("paid", lambda s: int((s <= 0).sum())),
            max_consecutive_nonpayment=("paid", max_consecutive_nonpayment),
            tenure_months=("month", "nunique"),
        )
        .reset_index()
    )
    agg12["payment_ratio_12m"] = agg12["paid_sum_12m"] / agg12["accrual_sum_12m"].replace(0, np.nan)
    agg12["payment_ratio_12m"] = agg12["payment_ratio_12m"].clip(lower=0, upper=2)
    agg12 = agg12.drop(columns=["accrual_sum_12m", "paid_sum_12m"])

    ratio3 = (
        win3.groupby("ls", dropna=False)
        .agg(accrual_sum_3m=("accrual", "sum"), paid_sum_3m=("paid", "sum"))
        .reset_index()
    )
    ratio3["payment_ratio_3m"] = ratio3["paid_sum_3m"] / ratio3["accrual_sum_3m"].replace(0, np.nan)
    ratio3["payment_ratio_3m"] = ratio3["payment_ratio_3m"].clip(lower=0, upper=2)
    ratio3 = ratio3[["ls", "payment_ratio_3m"]]

    trend6 = (
        win6.sort_values(["ls", "month"])
        .groupby("ls", dropna=False)["sz_start"]
        .apply(lambda s: linear_slope(s.to_numpy()))
        .rename("dz_trend_slope_6m")
        .reset_index()
    )
    vol6 = (
        win6.groupby("ls", dropna=False)["sz_start"]
        .std(ddof=0)
        .rename("dz_volatility_6m")
        .reset_index()
    )

    features = current.merge(agg12, on="ls", how="outer")
    features = features.merge(ratio3, on="ls", how="left")
    features = features.merge(trend6, on="ls", how="left")
    features = features.merge(vol6, on="ls", how="left")

    features["debt_to_accrual_ratio"] = features["DZ_current"] / features["avg_accrual_12m"].replace(0, np.nan)
    features["log1p_DZ_current"] = np.log1p(features["DZ_current"].clip(lower=0))
    return features


def load_payments(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "03 Оплаты ХК.csv"
    df = pd.read_csv(path, dtype="string", sep=None, engine="python")
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    df = df.rename(
        columns={
            "Номер": "ls_raw",
            "Дата оплаты": "payment_date_raw",
            "Сумма": "amount_raw",
            "Способ оплаты": "payment_method_raw",
        }
    )
    df["ls"] = normalize_ls(df["ls_raw"])
    df["payment_date"] = pd.to_datetime(df["payment_date_raw"], dayfirst=True, errors="coerce")
    df["amount"] = parse_money(df["amount_raw"])
    df["payment_method"] = df["payment_method_raw"].astype("string").str.strip().fillna("Unknown")
    return df.dropna(subset=["ls", "payment_date"])


def compute_payment_features(payments: pd.DataFrame, ctx: BuildContext) -> pd.DataFrame:
    win12 = payments[
        (payments["payment_date"] >= ctx.window12_start) & (payments["payment_date"] <= ctx.asof_date)
    ].copy()

    def payment_regularity(days: pd.Series) -> float:
        if days.shape[0] < 3:
            return np.nan
        ordered = days.sort_values().to_numpy()
        intervals = np.diff(ordered).astype("timedelta64[D]").astype(float)
        if intervals.size == 0:
            return np.nan
        std = float(np.std(intervals, ddof=0))
        return float(np.log1p(1.0 / (std + 1.0)))

    grouped = win12.groupby("ls", dropna=False)
    features = grouped.agg(
        payments_count_12m=("payment_date", "count"),
        median_payment_amount_12m=("amount", "median"),
        last_payment_date=("payment_date", "max"),
    ).reset_index()

    features["days_since_last_payment"] = (ctx.asof_date - features["last_payment_date"]).dt.days
    features = features.drop(columns=["last_payment_date"])

    regularity = grouped["payment_date"].apply(payment_regularity).rename("payment_regularity").reset_index()
    features = features.merge(regularity, on="ls", how="left")

    dominant_method = (
        grouped["payment_method"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown")
        .rename("dominant_payment_method")
        .reset_index()
    )
    return features.merge(dominant_method, on="ls", how="left")


def read_measure_events(path: Path, measure_name: str) -> pd.DataFrame:
    raw = pd.read_excel(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    if raw.shape[0] == 0:
        return pd.DataFrame(columns=["ls", "event_date", "measure"])

    first_row = [str(v).strip() for v in raw.iloc[0].tolist()]
    if len(first_row) >= 2 and first_row[0].lower() in {"лс", "номер"}:
        raw.columns = ["ls_raw", "event_date_raw"] + [f"extra_{i}" for i in range(raw.shape[1] - 2)]
        raw = raw.iloc[1:, :2].copy()
    else:
        raw = raw.iloc[:, :2].copy()
        raw.columns = ["ls_raw", "event_date_raw"]

    out = pd.DataFrame(
        {
            "ls": normalize_ls(raw["ls_raw"]),
            "event_date": pd.to_datetime(raw["event_date_raw"], errors="coerce"),
            "measure": measure_name,
        }
    )
    return out.dropna(subset=["ls", "event_date"])


def load_all_measure_events(data_dir: Path) -> pd.DataFrame:
    frames = []
    for measure, filename in MEASURE_FILES.items():
        path = data_dir / filename
        if path.exists():
            frames.append(read_measure_events(path, measure))

    if not frames:
        return pd.DataFrame(columns=["ls", "event_date", "measure"])

    events = pd.concat(frames, ignore_index=True).sort_values(["ls", "event_date"])
    return events


def compute_stage(events: pd.DataFrame, asof_date: pd.Timestamp) -> pd.DataFrame:
    stage_rank = {
        "autodial": 1,
        "email": 1,
        "sms": 1,
        "operator_call": 1,
        "claim": 1,
        "visit": 1,
        "restriction_notice": 2,
        "restriction": 3,
        "court_application": 4,
        "court_result": 4,
    }
    stage_name = {
        0: "None",
        1: "Информирование",
        2: "Уведомление_ограничения",
        3: "Ограничение",
        4: "Суд",
    }
    scoped = events[events["event_date"] <= asof_date].copy()
    scoped["rank"] = scoped["measure"].map(stage_rank).fillna(0)
    max_rank = scoped.groupby("ls", dropna=False)["rank"].max().rename("stage_rank").reset_index()
    max_rank["stage_reached"] = max_rank["stage_rank"].map(stage_name).fillna("None")
    return max_rank[["ls", "stage_reached"]]


def build_payment_index(payments: pd.DataFrame) -> dict[int, np.ndarray]:
    payment_index: dict[int, np.ndarray] = {}
    for ls, g in payments.groupby("ls", dropna=False):
        valid = g["payment_date"].dropna().sort_values().to_numpy(dtype="datetime64[ns]")
        if len(valid):
            payment_index[int(ls)] = valid
    return payment_index


def mark_event_paid_within_days(
    events: pd.DataFrame,
    payment_index: dict[int, np.ndarray],
    horizon_days: int = 14,
) -> pd.Series:
    paid_flags = np.zeros(len(events), dtype=int)
    end_delta = np.timedelta64(horizon_days, "D")

    ls_values = events["ls"].astype("Int64").to_numpy()
    event_values = events["event_date"].to_numpy(dtype="datetime64[ns]")

    for idx, (ls, event_dt) in enumerate(zip(ls_values, event_values)):
        if pd.isna(ls):
            continue
        payment_dates = payment_index.get(int(ls))
        if payment_dates is None or payment_dates.size == 0:
            continue
        left = np.searchsorted(payment_dates, event_dt, side="left")
        right = np.searchsorted(payment_dates, event_dt + end_delta, side="right")
        if right > left:
            paid_flags[idx] = 1
    return pd.Series(paid_flags, index=events.index)


def compute_measure_features(events: pd.DataFrame, payments: pd.DataFrame, ctx: BuildContext) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["ls"])

    scoped_all = events[events["event_date"] <= ctx.asof_date].copy()
    win12 = scoped_all[
        (scoped_all["event_date"] >= ctx.window12_start) & (scoped_all["event_date"] <= ctx.asof_date)
    ].copy()

    pivot_counts = (
        win12.assign(v=1)
        .pivot_table(index="ls", columns="measure", values="v", aggfunc="sum", fill_value=0)
        .reset_index()
    )
    rename_map = {
        "autodial": "n_autodial_12m",
        "email": "n_email_12m",
        "sms": "n_sms_12m",
        "operator_call": "n_operator_calls_12m",
        "visit": "n_visits_12m",
        "claim": "n_claims_12m",
        "restriction_notice": "n_restriction_notifications_12m",
        "restriction": "n_restrictions_12m",
        "court_application": "n_court_applications_12m",
        "court_result": "n_court_results_12m",
    }
    pivot_counts = pivot_counts.rename(columns=rename_map)

    for col in rename_map.values():
        if col not in pivot_counts.columns:
            pivot_counts[col] = 0

    totals = win12.groupby("ls", dropna=False).size().rename("n_measures_12m").reset_index()
    last_measure = (
        scoped_all.groupby("ls", dropna=False)["event_date"]
        .max()
        .rename("last_measure_date")
        .reset_index()
    )
    last_measure["days_since_last_measure"] = (ctx.asof_date - last_measure["last_measure_date"]).dt.days
    last_measure = last_measure[["ls", "days_since_last_measure"]]

    stage = compute_stage(scoped_all, ctx.asof_date)

    ever_restricted = (
        scoped_all.assign(flag=scoped_all["measure"].eq("restriction").astype(int))
        .groupby("ls", dropna=False)["flag"]
        .max()
        .replace({1: "Yes", 0: "No"})
        .rename("ever_restricted")
        .reset_index()
    )
    ever_in_court = (
        scoped_all.assign(
            flag=scoped_all["measure"].isin({"court_application", "court_result"}).astype(int)
        )
        .groupby("ls", dropna=False)["flag"]
        .max()
        .replace({1: "Yes", 0: "No"})
        .rename("ever_in_court")
        .reset_index()
    )

    payment_index = build_payment_index(payments[payments["payment_date"] <= ctx.asof_date])
    win12["paid_in_14d"] = mark_event_paid_within_days(win12, payment_index, horizon_days=14)

    informational = win12[win12["measure"].isin(INFORMATIONAL_MEASURES)]
    inf_rate = (
        informational.groupby("ls", dropna=False)["paid_in_14d"]
        .mean()
        .rename("response_rate_informational")
        .reset_index()
    )

    human = win12[win12["measure"].isin(HUMAN_INTENSIVE_MEASURES)]
    human_rate = (
        human.groupby("ls", dropna=False)["paid_in_14d"]
        .mean()
        .rename("response_rate_operator_or_visit")
        .reset_index()
    )

    features = totals.merge(pivot_counts, on="ls", how="outer")
    features = features.merge(last_measure, on="ls", how="left")
    features = features.merge(stage, on="ls", how="left")
    features = features.merge(ever_restricted, on="ls", how="left")
    features = features.merge(ever_in_court, on="ls", how="left")
    features = features.merge(inf_rate, on="ls", how="left")
    features = features.merge(human_rate, on="ls", how="left")
    return features


def build_context(data_dir: Path, snapshot_month: str | None) -> BuildContext:
    osv_long = load_osv_long(data_dir)
    if snapshot_month:
        snap = to_month_start(pd.Timestamp(snapshot_month))
    else:
        snap = to_month_start(osv_long["month"].max())
    asof_date = snap + pd.offsets.MonthEnd(0)

    return BuildContext(
        data_dir=data_dir,
        snapshot_month=snap,
        asof_date=asof_date,
        window12_start=snap - pd.DateOffset(months=11),
        window6_start=snap - pd.DateOffset(months=5),
        window3_start=snap - pd.DateOffset(months=2),
    )


def osv_inactive_ls_ids(osv_long: pd.DataFrame) -> set[int]:
    """ЛС без движения по ОСВ во всей выгрузке: сумма начислений, оплат и max СЗ — все нули."""

    def nz(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").fillna(0)

    grouped = (
        osv_long.dropna(subset=["ls"])
        .groupby("ls", dropna=False)
        .agg(
            accrual_sum=("accrual", lambda s: float(nz(s).sum())),
            paid_sum=("paid", lambda s: float(nz(s).sum())),
            sz_max=("sz_start", lambda s: float(nz(s).max())),
        )
    )
    mask = (grouped["accrual_sum"] == 0) & (grouped["paid_sum"] == 0) & (grouped["sz_max"] == 0)
    return {int(ls) for ls in grouped.loc[mask].index}


def fill_missing_for_kprototypes(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = [
        "has_phone",
        "has_ebill",
        "is_gasified",
        "has_benefits",
        "not_residing",
        "remote_disconnect_possible",
        "room_type",
        "meter_installation",
        "settlement_type",
        "stage_reached",
        "dominant_payment_method",
        "ever_restricted",
        "ever_in_court",
    ]
    numeric_cols = [c for c in df.columns if c not in {"ls"} | set(categorical_cols)]

    out = df.copy()
    for col in categorical_cols:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("Unknown")
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].fillna(out[col].median())
    return out


def build_features(
    data_dir: Path,
    snapshot_month: str | None,
    *,
    exclude_osv_inactive: bool = False,
    only_delinquent_snapshot: bool = False,
    warn_osv_inactive: bool = True,
) -> pd.DataFrame:
    ctx = build_context(data_dir, snapshot_month)

    general = load_general_info(data_dir)
    osv_long = load_osv_long(data_dir)
    inactive_ids = osv_inactive_ls_ids(osv_long)
    if warn_osv_inactive and inactive_ids:
        print(
            f"OSV: ЛС без начислений/оплат/сальдо во всей выгрузке (нулевая история): {len(inactive_ids)}",
            file=sys.stderr,
        )

    osv_features = compute_osv_features(osv_long, ctx)

    payments = load_payments(data_dir)
    payment_features = compute_payment_features(payments, ctx)

    events = load_all_measure_events(data_dir)
    measure_features = compute_measure_features(events, payments, ctx)

    ls_universe = pd.Series(
        pd.concat(
            [
                general["ls"],
                osv_long["ls"],
                payments["ls"],
                events["ls"] if not events.empty else pd.Series(dtype="Int64"),
            ],
            ignore_index=True,
        )
        .dropna()
        .astype("Int64")
        .unique(),
        name="ls",
    ).to_frame()

    features = ls_universe.merge(general, on="ls", how="left")
    features = features.merge(osv_features, on="ls", how="left")
    features = features.merge(payment_features, on="ls", how="left")
    features = features.merge(measure_features, on="ls", how="left")

    features["month_sin"] = np.sin(2 * np.pi * ctx.snapshot_month.month / 12.0)
    features["month_cos"] = np.cos(2 * np.pi * ctx.snapshot_month.month / 12.0)
    features["snapshot_month"] = ctx.snapshot_month
    features["asof_date"] = ctx.asof_date

    before = len(features)
    if exclude_osv_inactive and inactive_ids:
        features = features[~features["ls"].isin(inactive_ids)]
        print(
            f"Исключено ЛС с нулевой ОСВ: {before - len(features)} (осталось {len(features)})",
            file=sys.stderr,
        )

    if only_delinquent_snapshot:
        dz = pd.to_numeric(features["DZ_current"], errors="coerce")
        features = features[dz.fillna(0) > 0]
        print(
            f"Оставлены только ЛС с DZ_current > 0 на {ctx.snapshot_month.date()}: {len(features)} строк",
            file=sys.stderr,
        )

    features = fill_missing_for_kprototypes(features)
    return features.sort_values("ls").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build K-Prototypes feature table from case files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Материалы к кейсу"),
        help="Directory with source files 01..13.",
    )
    parser.add_argument(
        "--snapshot-month",
        type=str,
        default=None,
        help="Snapshot month (YYYY-MM-01). If omitted, max month from OSV is used.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("output/kprototypes_features.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("output/kprototypes_features.parquet"),
        help="Output Parquet path.",
    )
    parser.add_argument(
        "--exclude-osv-inactive",
        action="store_true",
        help="Исключить ЛС, у которых по всей ОСВ суммарно 0 начислений, 0 оплат и max(СЗ на начало)=0.",
    )
    parser.add_argument(
        "--only-delinquent",
        action="store_true",
        help="Оставить только ЛС с DZ_current > 0 на snapshot_month (до импутации пропусков).",
    )
    parser.add_argument(
        "--no-osv-inactive-warn",
        action="store_true",
        help="Не печатать предупреждение о числе ЛС с нулевой ОСВ.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)

    features = build_features(
        args.data_dir,
        args.snapshot_month,
        exclude_osv_inactive=args.exclude_osv_inactive,
        only_delinquent_snapshot=args.only_delinquent,
        warn_osv_inactive=not args.no_osv_inactive_warn,
    )
    features.to_csv(args.output_csv, index=False)

    print(f"Built features: {features.shape[0]} rows, {features.shape[1]} cols")
    print(f"CSV: {args.output_csv}")
    try:
        features.to_parquet(args.output_parquet, index=False)
        print(f"Parquet: {args.output_parquet}")
    except ImportError as exc:
        print(f"Parquet skipped ({exc.__class__.__name__}): {exc}")


if __name__ == "__main__":
    main()
