#!/usr/bin/env python3
"""
Пункт 2.2 стратегии: Прогнозирование отклика и uplift-моделирование.

Для каждой меры воздействия строит:
  - T-learner: mu1(X) = E[Y|T=1,X], mu0(X) = E[Y|T=0,X], uplift = mu1 - mu0
  - Propensity-score модель e(X) = P(T=1|X) для IPW / DR-коррекции
  - Матрицу utility(i,m) = uplift_pay(i,m) * expected_recovery(i,m) - cost(m) - risk_penalty(i,m)

Выходные файлы:
  output/response_train_<measure>.parquet  — обучающий датасет с таргетами
  output/response_model_summary.csv        — качество моделей (AUC, AUPR)
  output/utility_matrix.csv               — utility(i,m) по всем ЛС и мерам
  output/utility_matrix.parquet
  models/propensity_<measure>.joblib
  models/mu1_<measure>.joblib
  models/mu0_<measure>.joblib

Запуск:
  PYTHONPATH=/tmp/pylibs python3 scripts/build_response_models.py \\
      --data-dir "Материалы к кейсу" \\
      --features output/kprototypes_features.csv \\
      --labels   output/kprototypes_labels.csv \\
      --out-dir  output \\
      --models-dir models
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Конфигурация мер воздействия
# ---------------------------------------------------------------------------

MEASURE_META: Dict[str, dict] = {
    # cost_tier — порядковая шкала 1..5 по убыванию дешевизны (из Readme: Информирование ≪ Выезд ≪ Принудительные ≪ Суд).
    # Рублёвые cost намеренно не указаны: реальная себестоимость не предоставлена бизнесом.
    # Для оверрайда — см. configs/costs.yaml (опционально).
    "autodial": {
        "file": "04 Автодозвон ХК.xlsx",
        "cost_tier": 1,   # автоматизированное информирование, минимальные затраты
    },
    "sms": {
        "file": "06 СМС ХК.xlsx",
        "cost_tier": 1,
    },
    "email": {
        "file": "05 E-mail ХК.xlsx",
        "cost_tier": 1,
    },
    "operator_call": {
        "file": "07 Обзвон оператором ХК.xlsx",
        "cost_tier": 3,   # ручной труд оператора
    },
    "claim": {
        "file": "08 Претензия ХК.xlsx",
        "cost_tier": 3,   # бумажный документ + отправка
    },
    "visit": {
        "file": "09 Выезд к абоненту ХК.xlsx",
        "cost_tier": 4,   # выезд — высокие ресурсы
    },
    "restriction_notice": {
        "file": "10 Уведомление о введении ограничения ХК.xlsx",
        "cost_tier": 2,
    },
    "restriction": {
        "file": "11 Ограничение ХК.xlsx",
        "cost_tier": 4,   # принудительная мера
    },
    "court_application": {
        "file": "12 Заявление о выдаче судебного приказа ХК.xlsx",
        "cost_tier": 5,   # судебная процедура — максимальные затраты
    },
}

# Признаки для ML
CAT_FEATURES = [
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
    "ever_restricted",
    "ever_in_court",
    "dominant_payment_method",
]
NUM_FEATURES = [
    "DZ_current",
    "avg_accrual_12m",
    "avg_payment_12m",
    "months_with_full_payment_12m",
    "months_with_zero_payment_12m",
    "max_consecutive_nonpayment",
    "tenure_months",
    "payment_ratio_12m",
    "payment_ratio_3m",
    "dz_trend_slope_6m",
    "dz_volatility_6m",
    "debt_to_accrual_ratio",
    "log1p_DZ_current",
    "payments_count_12m",
    "median_payment_amount_12m",
    "days_since_last_payment",
    "payment_regularity",
    "n_measures_12m",
    "n_autodial_12m",
    "n_claims_12m",
    "n_court_applications_12m",
    "n_court_results_12m",
    "n_email_12m",
    "n_operator_calls_12m",
    "n_restrictions_12m",
    "n_restriction_notifications_12m",
    "n_sms_12m",
    "n_visits_12m",
    "days_since_last_measure",
    "response_rate_informational",
    "response_rate_operator_or_visit",
    "month_sin",
    "month_cos",
]

FEATURE_COLS = NUM_FEATURES + CAT_FEATURES

# Период для построения датасета (оставляем ≥30 дней до конца данных по оплатам)
TREATMENT_WINDOW_START = pd.Timestamp("2025-01-01")
TREATMENT_WINDOW_END = pd.Timestamp("2026-02-28")
FOLLOWUP_DAYS = 30

# LightGBM гиперпараметры (быстрые, но достаточные)
LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_features(features_path: Path, labels_path: Path) -> pd.DataFrame:
    """Загружает признаки и метки сегментации, объединяет в один датафрейм."""
    log.info("Загрузка признаков: %s", features_path)
    feats = pd.read_csv(features_path, low_memory=False)
    feats["ls"] = pd.to_numeric(feats["ls"], errors="coerce").astype("Int64")

    log.info("Загрузка меток: %s", labels_path)
    labels = pd.read_csv(labels_path)
    labels["ls"] = pd.to_numeric(labels["ls"], errors="coerce").astype("Int64")

    df = feats.merge(labels, on="ls", how="left")
    df["segment"] = df["segment"].fillna(-1).astype(int)

    # Приводим числовые признаки
    for col in NUM_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Категориальные — string
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("Unknown")

    log.info("Признаки загружены: %d ЛС", len(df))
    return df


def load_payments(data_dir: Path) -> pd.DataFrame:
    """Загружает историю оплат из CSV файла."""
    path = data_dir / "03 Оплаты ХК.csv"
    log.info("Загрузка оплат: %s", path)
    df = pd.read_csv(path, dtype="string", sep=None, engine="python")
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    df = df.rename(
        columns={
            "Номер": "ls_raw",
            "Дата оплаты": "payment_date_raw",
            "Сумма": "amount_raw",
        }
    )
    df["ls"] = pd.to_numeric(df["ls_raw"], errors="coerce").astype("Int64")
    df["payment_date"] = pd.to_datetime(df["payment_date_raw"], dayfirst=True, errors="coerce")
    amount_clean = (
        df["amount_raw"]
        .str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    df["amount"] = pd.to_numeric(amount_clean, errors="coerce").fillna(0.0)
    df = df.dropna(subset=["ls", "payment_date"])
    df = df[df["payment_date"] >= TREATMENT_WINDOW_START]
    log.info("Оплат загружено: %d строк", len(df))
    return df[["ls", "payment_date", "amount"]]


def load_measure_events(data_dir: Path, fname: str) -> pd.DataFrame:
    """Загружает историю применения меры из xlsx файла."""
    path = data_dir / fname
    df = pd.read_excel(path, header=1)
    df.columns = ["ls", "event_date"]
    df["ls"] = pd.to_numeric(df["ls"], errors="coerce").astype("Int64")
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["ls", "event_date"])
    df = df[
        (df["event_date"] >= TREATMENT_WINDOW_START)
        & (df["event_date"] <= TREATMENT_WINDOW_END)
    ]
    # Берём первое применение меры для каждого ЛС (дедупликация)
    df = df.sort_values("event_date").groupby("ls", as_index=False).first()
    return df


# ---------------------------------------------------------------------------
# Построение таргетов
# ---------------------------------------------------------------------------

def build_targets(
    events: pd.DataFrame,
    payments: pd.DataFrame,
    followup_days: int = FOLLOWUP_DAYS,
) -> pd.DataFrame:
    """
    Для каждого события (ls, event_date) вычисляет:
      pay_30d          : 1 если была хотя бы одна оплата в [event_date, event_date+30d]
      recovered_30d    : суммарная оплата за 30 дней после события
    """
    # Индексируем оплаты по ls для быстрого поиска
    pay_idx = payments.set_index("ls")

    results = []
    for row in events.itertuples(index=False):
        ls = row.ls
        t0 = row.event_date
        t1 = t0 + pd.Timedelta(days=followup_days)
        try:
            ls_pays = pay_idx.loc[ls]
            if isinstance(ls_pays, pd.Series):
                ls_pays = ls_pays.to_frame().T
            mask = (ls_pays["payment_date"] >= t0) & (ls_pays["payment_date"] <= t1)
            total = ls_pays.loc[mask, "amount"].sum()
        except KeyError:
            total = 0.0
        results.append({"ls": ls, "event_date": t0, "recovered_30d": float(total)})

    result_df = pd.DataFrame(results)
    result_df["pay_30d"] = (result_df["recovered_30d"] > 0).astype(int)
    return result_df


# ---------------------------------------------------------------------------
# Построение датасета: treatment + control
# ---------------------------------------------------------------------------

def build_train_dataset(
    measure_name: str,
    events: pd.DataFrame,
    features: pd.DataFrame,
    payments: pd.DataFrame,
    control_ratio: float = 2.0,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """
    Объединяет treated и control наблюдения в обучающий датасет.

    Treated  : ЛС из events с таргетами за 30 дней после события
    Control  : ЛС с DZ_current > 0, не попавшие в treated; таргеты за 30 дней
               после случайной даты из распределения событий (non-informative)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    treated_ls = set(events["ls"].dropna().astype(int))

    # Таргеты для treated
    treated_targets = build_targets(events, payments)
    treated = events.merge(treated_targets, on=["ls", "event_date"], how="left")
    treated["treatment"] = 1

    # Control: ЛС с долгом, которые не получили эту меру
    debtors = features[
        (features["DZ_current"].fillna(0) > 0)
        & (~features["ls"].isin(treated_ls))
    ][["ls"]].copy()

    n_control = min(int(len(treated) * control_ratio), len(debtors))
    if n_control == 0:
        log.warning("[%s] Нет контрольной группы!", measure_name)
        return treated.merge(features, on="ls", how="left")

    control_sample = debtors.sample(n=n_control, random_state=42).copy()

    # Назначаем случайные даты из распределения событий treated
    event_dates = events["event_date"].dropna().values
    ref_dates = rng.choice(event_dates, size=n_control, replace=True)
    control_sample["event_date"] = ref_dates
    control_sample["event_date"] = pd.to_datetime(control_sample["event_date"])

    # Таргеты для control
    control_targets = build_targets(control_sample, payments)
    control_sample = control_sample.merge(control_targets, on=["ls", "event_date"], how="left")
    control_sample["treatment"] = 0

    # Объединяем
    all_data = pd.concat([treated, control_sample], ignore_index=True)

    # Джойним с признаками
    all_data = all_data.merge(features, on="ls", how="left")

    # Заполняем пропуски в таргетах
    all_data["pay_30d"] = all_data["pay_30d"].fillna(0).astype(int)
    all_data["recovered_30d"] = all_data["recovered_30d"].fillna(0.0)

    log.info(
        "[%s] Dataset: treated=%d (pay_rate=%.1f%%), control=%d (pay_rate=%.1f%%)",
        measure_name,
        len(treated),
        100.0 * treated["pay_30d"].fillna(0).mean(),
        n_control,
        100.0 * control_sample["pay_30d"].fillna(0).mean(),
    )
    return all_data


# ---------------------------------------------------------------------------
# Препроцессинг для LightGBM
# ---------------------------------------------------------------------------

def prepare_X(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
    """Возвращает X (numpy), список числовых и категориальных столбцов."""
    num_cols = [c for c in NUM_FEATURES if c in df.columns]
    cat_cols = [c for c in CAT_FEATURES if c in df.columns]

    X_num = df[num_cols].values.astype(np.float32)

    # Кодируем категориальные признаки integer-encoding
    X_cat_parts = []
    for col in cat_cols:
        le = LabelEncoder()
        encoded = le.fit_transform(df[col].fillna("Unknown").astype(str))
        X_cat_parts.append(encoded.reshape(-1, 1))

    if X_cat_parts:
        X_cat = np.hstack(X_cat_parts).astype(np.float32)
    else:
        X_cat = np.empty((len(df), 0), dtype=np.float32)

    X = np.hstack([X_num, X_cat])
    all_cols = num_cols + cat_cols
    return X, num_cols, cat_cols


def make_lgb_dataset(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> lgb.Dataset:
    return lgb.Dataset(X, label=y, feature_name=feature_names, free_raw_data=False)


# ---------------------------------------------------------------------------
# T-learner
# ---------------------------------------------------------------------------

def fit_tlearner(
    train_df: pd.DataFrame,
    measure_name: str,
) -> Tuple[lgb.Booster, lgb.Booster, dict]:
    """
    Обучает два бустера:
      mu1 — на treated (T=1)
      mu0 — на control (T=0)
    Возвращает (mu1, mu0, metrics_dict).
    """
    all_cols = [c for c in FEATURE_COLS if c in train_df.columns]

    treated_df = train_df[train_df["treatment"] == 1].dropna(subset=all_cols)
    control_df = train_df[train_df["treatment"] == 0].dropna(subset=all_cols)

    if len(treated_df) < 20 or len(control_df) < 20:
        log.warning("[%s] Слишком мало данных (treated=%d, control=%d)", measure_name, len(treated_df), len(control_df))

    def _fit(subset: pd.DataFrame, name: str) -> Tuple[lgb.Booster, float, float]:
        X, num_cols, cat_cols = prepare_X(subset)
        y = subset["pay_30d"].values.astype(np.float32)
        feature_names = [c for c in FEATURE_COLS if c in subset.columns]

        if y.sum() < 5 or (y == 0).sum() < 5:
            log.warning("[%s/%s] Дисбаланс классов: pos=%d neg=%d", measure_name, name, int(y.sum()), int((y == 0).sum()))

        # CV для оценки метрик
        cv_probs = np.zeros(len(y))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx], feature_name=feature_names, free_raw_data=False)
            dval = lgb.Dataset(X[val_idx], label=y[val_idx], feature_name=feature_names, reference=dtrain, free_raw_data=False)
            model_cv = lgb.train(
                LGB_PARAMS,
                dtrain,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
            )
            cv_probs[val_idx] = model_cv.predict(X[val_idx])

        auc = roc_auc_score(y, cv_probs) if len(np.unique(y)) > 1 else 0.5
        aupr = average_precision_score(y, cv_probs) if len(np.unique(y)) > 1 else float(y.mean())

        # Финальная модель на всём подмножестве
        dtrain_full = lgb.Dataset(X, label=y, feature_name=feature_names, free_raw_data=False)
        model = lgb.train(
            {**LGB_PARAMS, "n_estimators": 400},
            dtrain_full,
            callbacks=[lgb.log_evaluation(-1)],
        )
        return model, auc, aupr

    log.info("[%s] Обучаем mu1 (treated, n=%d)...", measure_name, len(treated_df))
    mu1, auc1, aupr1 = _fit(treated_df, "mu1")

    log.info("[%s] Обучаем mu0 (control, n=%d)...", measure_name, len(control_df))
    mu0, auc0, aupr0 = _fit(control_df, "mu0")

    metrics = {
        "measure": measure_name,
        "n_treated": len(treated_df),
        "n_control": len(control_df),
        "pay_rate_treated": float(treated_df["pay_30d"].mean()),
        "pay_rate_control": float(control_df["pay_30d"].mean()),
        "naive_lift": float(treated_df["pay_30d"].mean() - control_df["pay_30d"].mean()),
        "auc_mu1": auc1,
        "aupr_mu1": aupr1,
        "auc_mu0": auc0,
        "aupr_mu0": aupr0,
    }
    return mu1, mu0, metrics


# ---------------------------------------------------------------------------
# Propensity model
# ---------------------------------------------------------------------------

def fit_propensity(
    train_df: pd.DataFrame,
    measure_name: str,
) -> Tuple[lgb.Booster, float]:
    """
    Обучает модель P(T=1 | X) для оценки склонности к получению меры.
    Возвращает (propensity_model, auc).
    """
    all_cols = [c for c in FEATURE_COLS if c in train_df.columns]
    df = train_df.dropna(subset=all_cols)
    X, _, _ = prepare_X(df)
    t = df["treatment"].values.astype(np.float32)
    feature_names = [c for c in FEATURE_COLS if c in df.columns]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probs = np.zeros(len(t))
    for tr_idx, val_idx in skf.split(X, t):
        dtrain = lgb.Dataset(X[tr_idx], label=t[tr_idx], feature_name=feature_names, free_raw_data=False)
        dval = lgb.Dataset(X[val_idx], label=t[val_idx], feature_name=feature_names, reference=dtrain, free_raw_data=False)
        m = lgb.train(
            LGB_PARAMS,
            dtrain,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )
        cv_probs[val_idx] = m.predict(X[val_idx])

    auc = roc_auc_score(t, cv_probs) if len(np.unique(t)) > 1 else 0.5

    # Полная модель
    dtrain_full = lgb.Dataset(X, label=t, feature_name=feature_names, free_raw_data=False)
    prop_model = lgb.train(
        {**LGB_PARAMS, "n_estimators": 400},
        dtrain_full,
        callbacks=[lgb.log_evaluation(-1)],
    )
    log.info("[%s] Propensity AUC=%.3f", measure_name, auc)
    return prop_model, auc


# ---------------------------------------------------------------------------
# Предсказание uplift на полном датасете
# ---------------------------------------------------------------------------

def predict_uplift(
    features: pd.DataFrame,
    mu1: lgb.Booster,
    mu0: lgb.Booster,
) -> np.ndarray:
    """Предсказывает uplift по формуле mu1(X) - mu0(X)."""
    df = features.copy()
    # Заполняем отсутствующие категориальные признаки
    for col in CAT_FEATURES:
        if col not in df.columns:
            df[col] = "Unknown"
        else:
            df[col] = df[col].fillna("Unknown").astype(str)

    X, _, _ = prepare_X(df)
    pred1 = mu1.predict(X)
    pred0 = mu0.predict(X)
    return pred1 - pred0


def predict_expected_recovery(
    features: pd.DataFrame,
    mu1: lgb.Booster,
) -> np.ndarray:
    """
    Оценивает ожидаемую сумму возврата: uplift_prob * avg_accrual.
    Используется как прокси для E[recovered_amount | T=1, X].
    """
    df = features.copy()
    for col in CAT_FEATURES:
        if col not in df.columns:
            df[col] = "Unknown"
        else:
            df[col] = df[col].fillna("Unknown").astype(str)

    X, _, _ = prepare_X(df)
    pay_prob = mu1.predict(X)

    avg_accrual = df["avg_accrual_12m"].fillna(0).clip(lower=0).values
    return pay_prob * avg_accrual


# ---------------------------------------------------------------------------
# Матрица expected_recovery_gain
# ---------------------------------------------------------------------------

def build_utility_matrix(
    features: pd.DataFrame,
    models: Dict[str, Tuple[lgb.Booster, lgb.Booster]],
) -> pd.DataFrame:
    """
    Строит матрицу скоров по мерам. Для каждой меры сохраняет:
      uplift_<m>                  : mu1(X) - mu0(X) — каузальный прирост вероятности оплаты
      expected_recovery_gain_<m>  : uplift × avg_accrual_12m — прирост ожидаемого возврата в руб. (из данных)
      cost_tier_<m>               : порядковая шкала ресурсоёмкости меры (1=дёшево, 5=суд)

    Рублёвая себестоимость мер не вычитается — реальные тарифы не предоставлены.
    Ранжирование мер: сначала по убыванию expected_recovery_gain, при равенстве — меньший cost_tier.
    """
    result = features[["ls", "segment"]].copy()
    avg_accrual = features["avg_accrual_12m"].fillna(0).clip(lower=0).values

    for measure_name, (mu1, mu0) in models.items():
        meta = MEASURE_META[measure_name]

        uplift = predict_uplift(features, mu1, mu0)
        # Прирост ожидаемого возврата: насколько больше вернём благодаря мере × средний месячный счёт
        expected_recovery_gain = uplift * avg_accrual

        result[f"uplift_{measure_name}"] = uplift.round(4)
        result[f"expected_recovery_gain_{measure_name}"] = expected_recovery_gain.round(2)
        result[f"cost_tier_{measure_name}"] = meta["cost_tier"]

    return result


# ---------------------------------------------------------------------------
# Основной пайплайн
# ---------------------------------------------------------------------------

def run(
    data_dir: Path,
    features_path: Path,
    labels_path: Path,
    out_dir: Path,
    models_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Загрузка признаков + меток
    features = load_features(features_path, labels_path)

    # 2. Загрузка оплат (единожды для всех мер)
    payments = load_payments(data_dir)

    all_metrics: List[dict] = []
    trained_models: Dict[str, Tuple[lgb.Booster, lgb.Booster]] = {}

    # 3. Цикл по мерам
    for measure_name, meta in MEASURE_META.items():
        log.info("=" * 60)
        log.info("Мера: %s (%s)", measure_name, meta["file"])

        # Загрузка событий меры
        try:
            events = load_measure_events(data_dir, meta["file"])
        except FileNotFoundError:
            log.warning("[%s] Файл не найден: %s — пропускаем", measure_name, meta["file"])
            continue

        if len(events) == 0:
            log.warning("[%s] Нет событий в окне — пропускаем", measure_name)
            continue

        # Построение датасета
        train_df = build_train_dataset(
            measure_name=measure_name,
            events=events,
            features=features,
            payments=payments,
        )

        # Сохраняем обучающий датасет
        train_path = out_dir / f"response_train_{measure_name}.parquet"
        save_cols = ["ls", "event_date", "treatment", "pay_30d", "recovered_30d", "segment"] + FEATURE_COLS
        save_cols = [c for c in save_cols if c in train_df.columns]
        train_df[save_cols].to_parquet(train_path, index=False)
        log.info("[%s] Датасет сохранён: %s (%d строк)", measure_name, train_path, len(train_df))

        # T-learner
        mu1, mu0, metrics = fit_tlearner(train_df, measure_name)
        all_metrics.append(metrics)
        trained_models[measure_name] = (mu1, mu0)

        # Propensity model
        prop_model, prop_auc = fit_propensity(train_df, measure_name)
        metrics["propensity_auc"] = prop_auc

        # Сохраняем модели
        joblib.dump(mu1, models_dir / f"mu1_{measure_name}.joblib")
        joblib.dump(mu0, models_dir / f"mu0_{measure_name}.joblib")
        joblib.dump(prop_model, models_dir / f"propensity_{measure_name}.joblib")

        log.info(
            "[%s] AUC mu1=%.3f, AUC mu0=%.3f, naive_lift=+%.1f%%",
            measure_name, metrics["auc_mu1"], metrics["auc_mu0"], 100 * metrics["naive_lift"],
        )

    # 4. Сводная таблица метрик
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = out_dir / "response_model_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log.info("\nСводка метрик сохранена: %s", metrics_path)
    log.info("\n%s", metrics_df.to_string(index=False))

    # 5. Матрица utility
    log.info("=" * 60)
    log.info("Строим матрицу utility(i,m)...")
    utility_df = build_utility_matrix(features, trained_models)

    utility_csv = out_dir / "utility_matrix.csv"
    utility_parquet = out_dir / "utility_matrix.parquet"
    utility_df.to_csv(utility_csv, index=False)
    utility_df.to_parquet(utility_parquet, index=False)
    log.info("Матрица utility сохранена: %s (%d ЛС × %d столбцов)", utility_csv, len(utility_df), len(utility_df.columns))

    # Быстрая статистика по expected_recovery_gain
    gain_cols = [c for c in utility_df.columns if c.startswith("expected_recovery_gain_")]
    log.info("\nСтатистика expected_recovery_gain по мерам (руб., из данных):")
    log.info("\n%s", utility_df[gain_cols].describe().round(2).to_string())

    log.info("\nГотово!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Построение response/uplift моделей (п.2.2)")
    p.add_argument(
        "--data-dir",
        default="Материалы к кейсу",
        help="Папка с исходными xlsx/csv файлами",
    )
    p.add_argument(
        "--features",
        default="output/kprototypes_features.csv",
        help="Файл признаков (CSV)",
    )
    p.add_argument(
        "--labels",
        default="output/kprototypes_labels.csv",
        help="Файл меток сегментов (CSV)",
    )
    p.add_argument(
        "--out-dir",
        default="output",
        help="Папка для выходных файлов",
    )
    p.add_argument(
        "--models-dir",
        default="models",
        help="Папка для сохранения моделей",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base = Path(__file__).parent.parent

    run(
        data_dir=base / args.data_dir,
        features_path=base / args.features,
        labels_path=base / args.labels,
        out_dir=base / args.out_dir,
        models_dir=base / args.models_dir,
    )
