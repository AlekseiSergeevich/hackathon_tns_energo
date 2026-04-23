# Пункт 2.2: Response/Uplift-моделирование

Скрипт `scripts/build_response_models.py` реализует прогнозирование отклика на меры воздействия согласно разделу 2.2 архитектурной стратегии.

## Что строится

Для каждой из 9 мер воздействия:

| Артефакт | Описание |
|---|---|
| `output/response_train_<мера>.parquet` | Обучающий датасет (treated + control) с таргетами |
| `models/mu1_<мера>.joblib` | LightGBM: `E[pay_30d \| T=1, X]` (treated) |
| `models/mu0_<мера>.joblib` | LightGBM: `E[pay_30d \| T=0, X]` (control) |
| `models/propensity_<мера>.joblib` | LightGBM: `P(T=1 \| X)` (selection bias diagnostic) |
| `output/response_model_summary.csv` | AUC, AUPR, pay_rate, naive_lift по мерам |
| `output/utility_matrix.parquet/.csv` | `utility(i,m)` = uplift × recovery − cost − risk для всех 136K ЛС |

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

Обучается модель `e(X) = P(T=1 | X)` для диагностики selection bias. Высокий AUC propensity (≥ 0.99 у всех мер) указывает на **сильный non-random treatment**: меры применялись не случайно — к тем, кто имел больший долг и хуже платил.

**Следствие**: наивный naive_lift отрицательный для всех мер (treated платят хуже, чем control), что является обычным явлением в задачах взыскания долгов. Для оценки истинного каузального эффекта необходим DR-learner / Causal Forest.

### Формула utility

```
utility(i, m) = uplift(i, m) × expected_recovery(i, m) - cost(m) - risk_penalty(i, m)
```

- `expected_recovery(i, m)` ≈ `mu1(X_i)` × `avg_accrual_12m_i`
- `cost(m)` — себестоимость меры (SMS: 1.5 руб., выезд: 600 руб., суд: 1200 руб.)
- `risk_penalty(i, m)` = `risk_base(m)` × `segment_factor(segment_i)` × `dz_penalty(DZ_i)`

Сегментный коэффициент риска:
| Сегмент | Описание | Фактор |
|---|---|---|
| 0 | Хроническая просрочка (Суд) | 1.5 |
| 1 | Средняя задолженность | 1.0 |
| 2 | Временные трудности | 0.8 |
| 3 | Незначительная задолженность | 0.5 |
| 4 | Без долга | 0.3 |
| 5 | Мелкая задержка | 0.4 |

## Запуск

```bash
pip install lightgbm pandas numpy scikit-learn joblib pyarrow openpyxl

python scripts/build_response_models.py \
    --data-dir "Материалы к кейсу" \
    --features output/kprototypes_features.csv \
    --labels   output/kprototypes_labels.csv \
    --out-dir  output \
    --models-dir models
```

Время выполнения: ~2-3 минуты на MacBook (M1/M2).

## Результаты (ключевые метрики)

| Мера | n_treated | AUC mu1 | AUC mu0 | Propensity AUC | Ср. utility (руб.) |
|---|---|---|---|---|---|
| autodial | 23 555 | 0.888 | 0.941 | 0.994 | -38.9 |
| **sms** | 13 998 | 0.893 | 0.953 | 1.000 | **+24.8** |
| email | 12 718 | 0.840 | 0.942 | 0.996 | -48.0 |
| operator_call | 9 938 | 0.911 | 0.922 | 0.999 | -13.3 |
| claim | 3 183 | 0.884 | 0.912 | 0.995 | -192.7 |
| visit | 5 281 | 0.968 | 0.941 | 1.000 | -601.6 |
| restriction_notice | 25 492 | 0.915 | 0.939 | 0.993 | -25.0 |
| restriction | 2 348 | 0.948 | 0.892 | 0.997 | -224.8 |
| court_application | 3 174 | 0.920 | 0.911 | 0.998 | -1252.8 |

**SMS** — единственная мера с положительной средней utility: дешевизна перевешивает низкий uplift.

## Ограничения и следующие шаги

1. **Selection bias**: Propensity AUC ≈ 1.0 → строгий DR-learner (EconML `LinearDRLearner`) улучшит оценки для дорогих мер.
2. **Одна фиче-матрица**: признаки — снапшот 2026-02-01. Для production нужны исторические снапшоты на момент каждого события.
3. **Следующий шаг — п.2.3 Scorer + п.3 Optimizer**: взять `utility_matrix.parquet`, применить фильтры Приложения №1, запустить MILP с лимитами Приложения №2.

## Jupyter notebook

Интерактивный анализ — `output/jupyter-notebook/response_uplift_analysis.ipynb`.
