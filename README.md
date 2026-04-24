# Система рекомендаций мер воздействия на должников — ТНС Энерго

ML-решение для персональной рекомендации мер взыскания задолженности.  
Реализует архитектурную стратегию `ml_strategy_debt_collection_c3e10d64.plan.md`.

---

## Быстрый старт

### Вариант 1 — GUI (рекомендуется)

```bash
pip install PySide6 pandas numpy scikit-learn lightgbm joblib pyarrow openpyxl pyyaml kmodes
python interface.py
```

Нажать кнопку **«Сформировать план мер на месяц (end-to-end)»**.

### Вариант 2 — CLI (4 шага)

```bash
pip install pandas numpy scikit-learn lightgbm joblib pyarrow openpyxl pyyaml kmodes

# Шаг 1: признаки ЛС
python scripts/build_kprototypes_features.py --data-dir "Материалы к кейсу"

# Шаг 2: сегментация K-Prototypes
python scripts/run_kprototypes_segmentation.py

# Шаг 3: uplift/response-модели (~3-5 мин)
python scripts/build_response_models.py --data-dir "Материалы к кейсу"

# Шаг 4: персональный план мер на месяц
python scripts/build_recommendations.py --month 2026-02
```

Результаты: `output/recommendations_2026-02.csv`, `output/recommendations_summary.csv`.

---

## Архитектура

```
Материалы к кейсу/                      ← исходные xlsx/csv (01..14)
scripts/
  build_kprototypes_features.py         ← шаг 1: 45 признаков на 136K ЛС
  run_kprototypes_segmentation.py       ← шаг 2: 6 сегментов (K-Prototypes)
  build_response_models.py              ← шаг 3: T-learner uplift + utility_matrix
  build_recommendations.py             ← шаг 4: фильтры + аллокатор + объяснения
configs/
  rules.yaml                           ← Приложение №1: допустимость мер
  limits.yaml                          ← Приложение №2: месячные лимиты
  costs.yaml                           ← cost_tier мер (опц. cost_rub при наличии тарифов)
output/
  kprototypes_features.parquet         ← признаки
  kprototypes_labels.csv               ← сегменты
  utility_matrix.parquet               ← uplift, expected_recovery_gain, cost_tier
  recommendations_YYYY-MM.csv          ← персональный план (ЛС, меры, объяснения)
  recommendations_summary.csv          ← агрегаты по мерам и лимитам
models/
  mu1_*.joblib, mu0_*.joblib           ← T-learner модели
  propensity_*.joblib                  ← propensity-score модели
docs/
  run_response_models.md               ← методология uplift-моделей
  run_recommendations.md               ← как запустить end-to-end, описание выхода
interface.py                           ← GUI (PySide6)
```

---

## Меры воздействия

| Мера | Файл | cost_tier |
|---|---|---|
| Автодозвон | 04 Автодозвон ХК.xlsx | 1 (фон) |
| E-mail | 05 E-mail ХК.xlsx | 1 (фон) |
| СМС | 06 СМС ХК.xlsx | 1 |
| Обзвон оператором | 07 Обзвон оператором ХК.xlsx | 3 |
| Претензия | 08 Претензия ХК.xlsx | 3 |
| Выезд к абоненту | 09 Выезд к абоненту ХК.xlsx | 4 |
| Уведомление об ограничении | 10 Уведомление о введении ограничения ХК.xlsx | 2 |
| Ограничение | 11 Ограничение ХК.xlsx | 4 |
| Заявление о судебном приказе | 12 Заявление о выдаче судебного приказа ХК.xlsx | 5 |

---

## Логика рекомендаций

1. **Фильтрация по Приложению №1** — срок ДЗ, сумма, наличие контактов, очередность стадий.
2. **State machine** — определяет, на какой стадии находится ЛС (информирование / уведомление / ограничение / суд), и разблокирует следующие меры.
3. **Combo builder** — фоновые меры (E-mail + Автодозвон) + лучшая ведущая мера по `expected_recovery_gain`.
4. **Жадный аллокатор** — назначение в рамках лимитов Приложения №2; «горячие» ЛС (сегменты 0, 1) обслуживаются в приоритете.
5. **Объяснения** — для каждого ЛС формируется текст с сегментом, сработавшими условиями, uplift и ERG.

### Метрика приоритизации

```
expected_recovery_gain(i, m) = uplift(i, m) × avg_accrual_12m(i)
```

- `uplift(i, m)` — каузальный прирост вероятности оплаты (T-learner: mu1 − mu0)
- `avg_accrual_12m` — средний месячный счёт ЛС (реальные данные)

Рублёвая себестоимость мер не включается — реальные тарифы не были предоставлены. При их получении добавить в `configs/costs.yaml` (блок `cost_rub`).

---

## Сегменты (K-Prototypes, k=6)

| Сегмент | Название | ЛС |
|---|---|---|
| 0 | Хроническая просрочка (в суде/ограничении) | 3 255 |
| 1 | Средняя задолженность (уведомление ограничения) | 7 380 |
| 2 | Временные трудности (высокое ДЗ, хорошая история) | 1 731 |
| 3 | Незначительная задолженность | 84 436 |
| 4 | Без задолженности | 26 283 |
| 5 | Мелкая задержка (только информирование) | 13 535 |

---

## Результаты (2026-02)

- Назначено мер: **17 809 ЛС** из 136 620 (13%)
- Покрытие «горячих» сегментов 0+1: **45–53%**
- Лимиты: СМС 100%, Выезд 100%, Ограничение 100%, Автодозвон 84%

Подробнее: [`docs/run_recommendations.md`](docs/run_recommendations.md) | [`docs/run_response_models.md`](docs/run_response_models.md)
