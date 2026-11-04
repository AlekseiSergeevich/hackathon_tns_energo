"""Обучающая выборка: срезы «ЛС-должник × месяц», признаки и таргет.

Срез M — состояние на 01.M. Признаки берутся только из колонок `start` месяца M
и из любых колонок месяцев ≤ M−1 (см. known_at в docs/interfaces.md).
Таргет — pay30_m месяца M: оплачено ≥ 30 % dz до конца M.

В выборку попадают только должники: dz ≥ P-01 и dz_overdue > 0.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Деление по месяцам. Holdout открываем один раз, при выборе модели (ML-7).
SPLITS = {
    "train": ["2025-07", "2025-08", "2025-09", "2025-10"],
    "valid": ["2025-11"],
    "test": ["2025-12", "2026-01", "2026-02"],
    "holdout": ["2026-03"],
}
SPLIT_OF_MONTH = {pd.Timestamp(m): s for s, ms in SPLITS.items() for m in ms}
ALL_MONTHS = [pd.Timestamp(m) for ms in SPLITS.values() for m in ms]

# Окно истории для признаков. Для среза 2025-07 доступно ровно 6 месяцев.
WINDOW = 6

# Колонки панели по known_at из interfaces.md.
START_COLS = [
    "saldo_start", "dz", "dz_overdue", "overdue_m", "overdue_censored",
    "months_no_payment", "months_no_payment_censored", "norm", "norm_source",
]
END_COLS = ["accrual", "paid", "paid_payments", "pay30_m"]

LS_INFO_FLAGS = ["has_phone", "has_ebill", "has_benefits", "not_residing", "remote_disconnect", "gasified"]

ID_COLS = ["ls", "month", "split"]
TARGET_COLS = ["y", "y_overdue", "y_any", "paid_m"]

# TODO(ORG-1): читать result.share из configs/settings.yaml.
RESULT_SHARE = 0.30


# ---------------------------------------------------------------- параметры из таблицы правил

def load_rule_param(rules_path: Path, code: str) -> str:
    """Значение параметра P-xx с листа «Параметры» таблицы правил."""
    params = pd.read_excel(rules_path, sheet_name="Параметры")
    row = params.loc[params["Код"].astype(str).str.strip() == code, "Значение"]
    if row.empty:
        raise ValueError(f"В таблице правил «{Path(rules_path).name}» нет параметра {code}.")
    return row.iloc[0]


def min_debt(rules_path: Path) -> float:
    """P-01: минимальная ДЗ для начала работы с ЛС."""
    return float(load_rule_param(rules_path, "P-01"))


# ---------------------------------------------------------------- признаки

def _month_index(s: pd.Series) -> pd.Series:
    return s.dt.year * 12 + s.dt.month


def _f64(s: pd.Series) -> pd.Series:
    """boolean/Int64/Float64 → float64, <NA> → NaN."""
    return pd.Series(s.astype("Float64").to_numpy(dtype="float64", na_value=np.nan), index=s.index, name=s.name)


def _to_float(df: pd.DataFrame) -> pd.DataFrame:
    """Все колонки, кроме категорий, → float64 (NaN = неизвестно): так их понимают LightGBM и sklearn."""
    out = df.copy()
    for c in out.columns:
        if not isinstance(out[c].dtype, pd.CategoricalDtype) and out[c].dtype != "float64":
            out[c] = _f64(out[c])
    return out


def population(panel: pd.DataFrame, month: pd.Timestamp, min_dz: float) -> pd.DataFrame:
    """Должники на 01.M: dz ≥ P-01 и есть просроченная часть. Только колонки start."""
    now = panel.loc[panel["month"] == month, ["ls", *START_COLS]]
    return now[(now["dz"] >= min_dz) & (now["dz_overdue"] > 0)]


def build_features(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    ls_info: pd.DataFrame,
    month: pd.Timestamp,
    min_dz: float,
    window: int = WINDOW,
) -> pd.DataFrame:
    """Признаки среза M. Читает только start-колонки M и всё, что было до M.

    window — сколько прошлых месяцев смотрим (не меньше 3). Суффикс признаков
    «_<window>m» показывает длину окна, например n_pay_months_6m.
    """
    if window < 3:
        raise ValueError(f"Окно истории должно быть не меньше 3 месяцев, а задано {window}.")
    month = pd.Timestamp(month)
    sfx = f"{window}m"
    now = population(panel, month, min_dz).set_index("ls").drop(columns=["norm_source", "saldo_start"])
    ls = now.index
    m_idx = month.year * 12 + month.month

    # --- финансы: прошлые месяцы M−1 … M−window в «широком» виде (колонка = лаг)
    past = panel[(panel["month"] < month) & panel["ls"].isin(ls)]
    past = past.assign(lag=m_idx - _month_index(past["month"]))
    past = past[past["lag"] <= window]
    vals = ["dz", "accrual", "paid_payments", "pay30_m"]
    past = past.assign(**{v: _f64(past[v]) for v in vals})
    wide = past.pivot(index="ls", columns="lag", values=vals).reindex(ls)
    lags = list(range(1, window + 1))
    w = {v: wide[v].reindex(columns=lags) for v in vals}
    last3 = [1, 2, 3]

    f = pd.DataFrame(index=ls)
    f["dz"] = now["dz"]
    f["log_dz"] = np.log1p(now["dz"])
    f["dz_overdue"] = now["dz_overdue"]
    f["overdue_share"] = now["dz_overdue"] / now["dz"]
    f["overdue_m"] = now["overdue_m"]
    f["overdue_censored"] = now["overdue_censored"]
    f["months_no_payment"] = now["months_no_payment"]
    f["months_no_payment_censored"] = now["months_no_payment_censored"]
    f["norm"] = now["norm"]
    f["dz_to_norm"] = now["dz"] / now["norm"].where(now["norm"] > 0)

    f["dz_lag1"] = w["dz"][1]
    f["dz_change_1m"] = now["dz"] - w["dz"][1]
    f["dz_change_3m"] = now["dz"] - w["dz"][3]
    f[f"dz_max_{sfx}"] = w["dz"].max(axis=1)
    f[f"n_debt_months_{sfx}"] = (w["dz"] >= min_dz).sum(axis=1)

    acc3 = w["accrual"][last3].sum(axis=1, min_count=1)
    acc_w = w["accrual"].sum(axis=1, min_count=1)
    pay3 = w["paid_payments"][last3].sum(axis=1, min_count=1)
    pay_w = w["paid_payments"].sum(axis=1, min_count=1)
    f["accrual_lag1"] = w["accrual"][1]
    f[f"accrual_mean_{sfx}"] = w["accrual"].mean(axis=1)
    f["paid_lag1"] = w["paid_payments"][1]
    f["paid_sum_3m"] = pay3
    f[f"paid_sum_{sfx}"] = pay_w
    f["pay_ratio_3m"] = pay3 / acc3.where(acc3 > 0)
    f[f"pay_ratio_{sfx}"] = pay_w / acc_w.where(acc_w > 0)
    f[f"n_pay_months_{sfx}"] = (w["paid_payments"] > 0).sum(axis=1)
    f["paid_lag1_to_dz"] = w["paid_payments"][1] / w["dz"][1].where(w["dz"][1] > 0)

    f["pay30_lag1"] = w["pay30_m"][1]
    f["pay30_lag2"] = w["pay30_m"][2]
    f["pay30_lag3"] = w["pay30_m"][3]
    f[f"pay30_rate_{sfx}"] = w["pay30_m"].mean(axis=1)

    # --- меры за M−1 … M−window
    ev = events[(events["month"] < month) & events["ls"].isin(ls)]
    ev = ev.assign(lag=m_idx - _month_index(ev["month"]))
    ev = ev[ev["lag"] <= window]
    measures = list(events["measure"].cat.categories)
    cnt = ev.pivot_table(index="ls", columns="measure", values="lag", aggfunc="size", observed=False)
    since = ev.pivot_table(index="ls", columns="measure", values="lag", aggfunc="min", observed=False)
    cnt = cnt.reindex(index=ls, columns=measures).fillna(0)
    since = since.reindex(index=ls, columns=measures)
    for m in measures:
        f[f"n_{m}_{sfx}"] = cnt[m]
        f[f"months_since_{m}"] = since[m]  # NaN = не было за окно
    f[f"n_measures_{sfx}"] = cnt.sum(axis=1)
    f["months_since_any_measure"] = since.min(axis=1)

    # --- анкета (статичный снимок: для старых срезов слегка «из будущего»)
    info = ls_info.set_index("ls").reindex(ls)
    for c in LS_INFO_FLAGS:
        f[c] = info[c]
    f["house_type"] = info["house_type"]

    f = _to_float(f)
    f.insert(0, "month", month)
    return f.reset_index()


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS + TARGET_COLS]


# ---------------------------------------------------------------- таргет

def attach_target(snap: pd.DataFrame, panel: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    """Таргет месяца M (end-колонки) — отдельно от признаков."""
    cur = panel.loc[panel["month"] == month, ["ls", "dz", "dz_overdue", "paid_payments", "pay30_m"]]
    cur = cur.set_index("ls").reindex(snap["ls"])
    out = snap.copy()
    out["paid_m"] = cur["paid_payments"].to_numpy()
    out["y"] = _f64(cur["pay30_m"]).to_numpy()
    out["y_overdue"] = (cur["paid_payments"] >= RESULT_SHARE * cur["dz_overdue"]).astype("float64").to_numpy()
    out["y_any"] = (cur["paid_payments"] > 0).astype("float64").to_numpy()
    return out


def build_dataset(
    tables: dict[str, pd.DataFrame], min_dz: float, months=None, window: int = WINDOW
) -> pd.DataFrame:
    """Все срезы из SPLITS (или months) в одну таблицу с колонкой split.

    Месяцы вне SPLITS получают split = <NA>.
    """
    months = [pd.Timestamp(m) for m in (months or ALL_MONTHS)]
    panel, events, ls_info = tables["panel"], tables["measure_events"], tables["ls_info"]
    parts = []
    for m in months:
        snap = build_features(panel, events, ls_info, m, min_dz, window)
        parts.append(attach_target(snap, panel, m))
        log.info("Срез %s: %d строк", f"{m:%Y-%m}", len(snap))
    df = pd.concat(parts, ignore_index=True)
    df.insert(2, "split", df["month"].map(SPLIT_OF_MONTH).astype("string"))
    df["house_type"] = df["house_type"].astype(pd.CategoricalDtype(ls_info["house_type"].cat.categories))
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Срез · split · строк · доли таргетов."""
    return (
        df.groupby("month")
        .agg(split=("split", "first"), rows=("ls", "size"), y=("y", "mean"),
             y_overdue=("y_overdue", "mean"), y_any=("y_any", "mean"), dz_median=("dz", "median"))
        .round(3)
    )


# ---------------------------------------------------------------- проверки на утечку (ML-3)

def cut_to_known(panel: pd.DataFrame, events: pd.DataFrame, month: pd.Timestamp):
    """Оставить только то, что известно на 01.M: месяцы < M, у M — только start-колонки."""
    p = panel[panel["month"] <= month].copy()
    for c in END_COLS:
        p.loc[p["month"] == month, c] = pd.NA if p[c].dtype.name in {"boolean", "Int64"} else np.nan
    e = events[events["month"] < month]
    return p, e


def check_no_leakage(tables: dict[str, pd.DataFrame], month: pd.Timestamp, min_dz: float) -> None:
    """Признаки среза M не меняются, если удалить всё, что стало известно после 01.M."""
    month = pd.Timestamp(month)
    panel, events, ls_info = tables["panel"], tables["measure_events"], tables["ls_info"]
    full = build_features(panel, events, ls_info, month, min_dz)
    p, e = cut_to_known(panel, events, month)
    cut = build_features(p, e, ls_info, month, min_dz)
    pd.testing.assert_frame_equal(full, cut, check_dtype=False)


def check_panel_start_cols(tables: dict[str, pd.DataFrame], month: pd.Timestamp) -> None:
    """start-колонки панели на 01.M не зависят от данных после 01.M.

    Пересобирает прото-панель из ОСВ до M и оплат до 01.M. Когда появится
    настоящая панель, этот тест переедет в tns/data.
    """
    from tns.ml.proto_panel import build_panel

    month = pd.Timestamp(month)
    panel, payments = tables["panel"], tables["payments"]
    osv = panel.loc[panel["month"] <= month, ["ls", "month", "saldo_start", "accrual", "paid"]].copy()
    osv.loc[osv["month"] == month, ["accrual", "paid"]] = np.nan
    rebuilt = build_panel(osv, payments[payments["payment_date"] < month])

    key = ["ls", "month"]
    a = panel.loc[panel["month"] == month, key + START_COLS].reset_index(drop=True)
    b = rebuilt.loc[rebuilt["month"] == month, key + START_COLS].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
