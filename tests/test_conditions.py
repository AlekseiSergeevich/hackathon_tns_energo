"""Тесты вычисления формальных условий таблицы правил (tns/rules/conditions.py)."""

from pathlib import Path

import numpy as np
import pytest

from tns.rules.conditions import ConditionError, Env, compile_condition, params_of
from tns.rules.loader import load_rulebook

BOOK = Path(__file__).resolve().parents[1] / "docs" / "Правила воздействия ТНС v0.3.xlsx"
PARAMS = {"P-01": 500.0, "P-06": 20.0, "P-08": 2.0, "P-09": 3.0, "P-02": None}
TODAY = np.datetime64("2025-05-01")


def env(size, **fields):
    results = fields.pop("results", {})
    exceptions = fields.pop("exceptions", None)
    return Env(fields=fields, results=results, today=TODAY, size=size, exceptions=exceptions)


def ev(text, e, params=PARAMS):
    return compile_condition(text, params).evaluate(e).tolist()


def test_all_book_conditions_compile():
    book = load_rulebook(BOOK)
    params = params_of(book)
    for t in book.transitions:
        compile_condition(t.condition, params, t.code)
    for r in book.rules.values():
        if r.condition:
            compile_condition(r.condition, params, r.code)


def test_and_with_params():
    e = env(4, dz=np.array([300, 900, 900, np.nan]), overdue_m=np.array([1, 0, 2, 3.0]))
    assert ev("dz >= P-01 AND overdue_m >= 1", e) == [False, False, True, False]


def test_precedence_and_or_not():
    e = env(3, a=np.array([True, False, False]), b=np.array([False, True, False]), c=np.array([True, True, False]))
    assert ev("a OR b AND c", e) == [True, True, False]  # AND сильнее OR
    assert ev("(a OR b) AND NOT c", e) == [False, False, False]
    assert ev("NOT a AND NOT b", e) == [False, False, True]


def test_arithmetic_and_empty_norm():
    e = env(3, dz=np.array([2900, 2000, 5000.0]), norm=np.array([1100, 1100, np.nan]))
    # пустой норматив — условие не выполнено
    assert ev("dz > P-08 * norm", e) == [True, False, False]


def test_is_null():
    dates = np.array(["2025-03-26", "NaT"], dtype="datetime64[ns]")
    e = env(2, notice_date=dates)
    assert ev("notice_date IS NULL", e) == [False, True]
    assert ev("notice_date IS NOT NULL", e) == [True, False]


def test_days_since_and_result():
    dates = np.array(["2025-03-26", "2025-04-20", "NaT"], dtype="datetime64[ns]")
    e = env(3, notice_date=dates, results={"notice": np.array([False, False, False])})
    assert ev("days_since(notice_date) >= P-06 AND NOT result(notice)", e) == [True, False, False]


def test_months_since_measure_alias():
    claims = np.array(["2025-01-10", "2025-03-01", "NaT"], dtype="datetime64[ns]")
    e = env(3, last_claim_date=claims)
    assert ev("months_since(claim) >= P-09", e) == [True, False, False]


def test_exceptions_default_false():
    e = env(2, dz=np.array([1.0, 2.0]))
    assert ev("NOT EX-05", e) == [True, True]
    assert ev("any(EX-*)", e) == [False, False]
    e2 = env(2, exceptions={"EX-05": np.array([True, False])})
    assert ev("any(EX-*)", e2) == [True, False]


def test_scalar_field_broadcasts():
    e = env(3, contract_active=True)
    assert ev("contract_active", e) == [True, True, True]


def test_unknown_param():
    with pytest.raises(ConditionError, match="нет параметра P-99"):
        compile_condition("dz > P-99", PARAMS)


def test_text_param_cannot_be_used():
    with pytest.raises(ConditionError, match="P-02 не число"):
        compile_condition("overdue_m > P-02", PARAMS)


def test_missing_field_message():
    with pytest.raises(ConditionError, match="нет данных для поля «has_mobile»"):
        compile_condition("has_mobile", PARAMS, "правило R-SMS-01").evaluate(env(1))


def test_unsupported_function():
    with pytest.raises(ConditionError, match="entry_threshold"):
        compile_condition("dz < entry_threshold(stage)", PARAMS).evaluate(env(1, dz=np.array([1.0]), stage=np.array([1])))


def test_syntax_errors():
    with pytest.raises(ConditionError):
        compile_condition("dz >= AND", PARAMS)
    with pytest.raises(ConditionError, match="непонятный символ"):
        compile_condition("dz ≥ 500", PARAMS)
    with pytest.raises(ConditionError, match="лишний текст"):
        compile_condition("dz > 1 2", PARAMS)
