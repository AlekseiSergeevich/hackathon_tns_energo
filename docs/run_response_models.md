# Пункт 2.2: Response/Uplift-моделирование

Скрипт `scripts/build_response_models.py` реализует прогнозирование отклика на меры воздействия согласно разделу 2.2 архитектурной стратегии.

## Что строится

Для каждой из 9 мер воздействия:

| Артефакт | Описание |
|---|---|
| `output/response_train_<мера>.parquet` | Обучающий датасет (treated + control) с таргетами |
| `models/mu1_<мера>.joblib` | LightGBM: `E[pay_30d \| T=1, X]` (treated) |
| `models/mu0_<мера>.joblib` | LightGBM: `E[pay_30d \| T=0, X]` (control) |
| `models/propensity_<мера>.joblib` | LightGBM: `P(T=1 \| X)` (диагностика selection bias) |
| `output/response_model_summary.csv` | AUC, pay_rate, naive_lift по мерам |
| `output/utility_matrix.parquet/.csv` | `uplift(i,m)`, `expected_recovery_gain(i,m)`, `cost_tier(m)` для 136K ЛС |

## Методология

### T-learner (meta-learner)

```
mu1(X) = E[Y | T=1, X]  ← LightGBM на treated-наблюдениях
mu0(X) = E[Y | T=0, X]  ← LightGBM на control-наблюдениях
uplift(X) = mu1(X) - mu0(X)
```

- **Y = `pay_30d`**: бинарный признак «оплатил в течение 30 дней после события»
- **X**: 45 числовых и категориальных признаков из `kprototypes_features.csv`
- **Treated**: ЛС, получившие меру в период 2025-01-01 — 2026-02-28 (первое событие)
- **Control**: ЛС с ДЗ > 0, не получившие данную меру; референсная дата — случайная из распределения дат treated

### Propensity score

Обучается модель `e(X) = P(T=1 | X)` для диагностики selection bias. Высокий AUC propensity (≥ 0.99) указывает на **сильный non-random treatment**: меры применялись не случайно — к тем, кто имел больший долг и хуже платил.

**Следствие**: наивный naive_lift отрицательный для всех мер (treated платят хуже, чем control), что является типичным явлением в задачах взыскания. Истинный каузальный эффект оценивается через T-learner uplift.

### Матрица utility (без ложной монетизации)

Вместо рублёвой функции `uplift × recovery − cost − risk` (где `cost` — выдуманные числа) используется прозрачная тройка показателей:

| Колонка | Формула | Источник |
|---|---|---|
| `uplift_<m>` | `mu1(X) − mu0(X)` | T-learner |
| `expected_recovery_gain_<m>` | `uplift × avg_accrual_12m` | данные (реальные рубли) |
| `cost_tier_<m>` | порядковый ранг 1..5 | экспертная шкала (Readme.docx) |

**Рублёвая себестоимость мер не вычитается** — реальные тарифы не предоставлены. При получении тарифов от бизнеса их можно добавить в `configs/costs.yaml` (блок `cost_rub`).

## Запуск

```bash
pip install lightgbm pandas numpy scikit-learn joblib pyarrow openpyxl pyyaml

python scripts/build_response_models.py \
    --data-dir "Материалы к кейсу" \
    --features output/kprototypes_features.csv \
    --labels   output/kprototypes_labels.csv \
    --out-dir  output \
    --models-dir models
```

Время выполнения: ~3-5 минут на MacBook (M1/M2).

## Результаты (ключевые метрики)

| Мера | n_treated | AUC mu1 | AUC mu0 | Propensity AUC | Mean uplift | Mean ERG (руб.) | cost_tier |
|---|---|---|---|---|---|---|---|
| autodial | 23 555 | 0.888 | 0.941 | 0.994 | −0.072 | −61.5 | 1 |
| **sms** | 13 998 | 0.893 | 0.953 | 1.000 | **+0.087** | **+23.0** | 1 |
| email | 12 718 | 0.840 | 0.942 | 0.996 | −0.109 | −91.6 | 1 |
| operator_call | 9 938 | 0.911 | 0.922 | 0.999 | +0.111 | +50.4 | 3 |
| claim | 3 183 | 0.884 | 0.912 | 0.995 | −0.275 | −192.5 | 3 |
| visit | 5 281 | 0.968 | 0.941 | 1.000 | −0.019 | +14.7 | 4 |
| restriction_notice | 25 492 | 0.915 | 0.939 | 0.993 | +0.110 | +44.3 | 2 |
| restriction | 2 348 | 0.948 | 0.892 | 0.997 | +0.155 | +80.7 | 4 |
| court_application | 3 174 | 0.920 | 0.911 | 0.998 | +0.006 | +9.5 | 5 |

- **Mean ERG** (`expected_recovery_gain`) = mean uplift × mean(avg_accrual_12m) — реальные рубли из данных.  
- Отрицательный ERG означает, что T-learner оценивает отрицательный uplift (мера снижает вероятность оплаты по сравнению с контролем). Это артефакт selection bias — те, кому применяли меру, изначально платили хуже.
- **SMS** и **Обзвон оператором** имеют наилучшее сочетание положительного uplift и низкого cost_tier.

## Ограничения и следующие шаги

1. **Selection bias**: Propensity AUC ≈ 1.0 → DR-learner (EconML `LinearDRLearner`) даст более точные оценки.
2. **Одна фиче-матрица**: признаки — снапшот 2026-02-01. Для production нужны исторические снапшоты.
3. **Следующий шаг**: `scripts/build_recommendations.py` — применяет фильтры Приложения №1, state machine, жадный аллокатор с лимитами Приложения №2 и формирует персональный план мер.

## Jupyter notebook

Интерактивный анализ — `output/jupyter-notebook/response_uplift_analysis.ipynb`.
