"""Вычисление формальных условий из таблицы правил.

Условие из книги, например «dz >= P-01 AND overdue_m >= 1», разбирается один раз
(compile_condition) и потом вычисляется сразу для всех ЛС месяца (Condition.evaluate):
поля — массивы numpy одинаковой длины, результат — массив True/False.

Что понимает разбор:
- AND, OR, NOT, скобки;
- сравнения >=, <=, >, <, =, !=; арифметика * / + -;
- числа, параметры P-xx (значение с листа «Параметры»), исключения EX-xx и any(EX-*);
- «поле IS NULL», «поле IS NOT NULL»;
- функции: result(предмет), days_since(дата), months_since(дата или мера).

Пустые значения: сравнение с пустым (NaN, NaT) даёт «Нет». Например, если норматив
не определён, условие «dz > P-08 * norm» не выполняется.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

KEYWORDS = {"AND", "OR", "NOT", "IS", "NULL"}

_TOKEN = re.compile(
    r"""\s*(?:
        (?P<num>\d+(?:[.,]\d+)?)
       |(?P<param>P-\d{2})
       |(?P<excany>EX-\*)
       |(?P<exc>EX-\d{2})
       |(?P<op>>=|<=|!=|<>|==|=|>|<|\*|/|\+|-)
       |(?P<lp>\()
       |(?P<rp>\))
       |(?P<comma>,)
       |(?P<word>[A-Za-z_][A-Za-z0-9_]*)
    )""",
    re.VERBOSE,
)

_CMP = {
    ">=": np.greater_equal, "<=": np.less_equal, ">": np.greater, "<": np.less,
    "=": np.equal, "==": np.equal, "!=": np.not_equal, "<>": np.not_equal,
}
_ARITH = {"*": np.multiply, "/": np.divide, "+": np.add, "-": np.subtract}


class ConditionError(ValueError):
    """Условие не удалось разобрать или вычислить. Сообщение — по-русски, с текстом условия."""


# ---------------------------------------------------------------------------
# Разбор
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens, pos = [], 0
    text = text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise ConditionError(f"непонятный символ «{text[pos:].strip()[:15]}» в условии «{text}»")
        kind = m.lastgroup
        value = m.group(kind)
        if kind == "word" and value.upper() in KEYWORDS:
            kind, value = "kw", value.upper()
        tokens.append((kind, value))
        pos = m.end()
    return tokens


class _Parser:
    """Рекурсивный спуск. Приоритет: OR < AND < NOT < сравнение < + - < * /."""

    def __init__(self, text: str):
        self.text = text
        self.tokens = _tokenize(text)
        self.i = 0

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self, kind: str | None = None, value: str | None = None) -> tuple[str, str]:
        tok = self.peek()
        if tok is None or (kind and tok[0] != kind) or (value and tok[1] != value):
            want = value or kind or "продолжение"
            got = tok[1] if tok else "конец условия"
            raise ConditionError(f"в условии «{self.text}» ожидалось «{want}», а встретилось «{got}»")
        self.i += 1
        return tok

    def accept(self, kind: str, value: str | None = None) -> bool:
        tok = self.peek()
        if tok and tok[0] == kind and (value is None or tok[1] == value):
            self.i += 1
            return True
        return False

    def parse(self) -> tuple:
        node = self.or_()
        if self.peek() is not None:
            raise ConditionError(f"в условии «{self.text}» лишний текст после «{self.tokens[self.i - 1][1]}»")
        return node

    def or_(self):
        node = self.and_()
        while self.accept("kw", "OR"):
            node = ("or", node, self.and_())
        return node

    def and_(self):
        node = self.not_()
        while self.accept("kw", "AND"):
            node = ("and", node, self.not_())
        return node

    def not_(self):
        if self.accept("kw", "NOT"):
            return ("not", self.not_())
        return self.cmp()

    def cmp(self):
        left = self.sum_()
        tok = self.peek()
        if tok and tok[0] == "kw" and tok[1] == "IS":
            self.i += 1
            negate = self.accept("kw", "NOT")
            self.take("kw", "NULL")
            return ("isnull", left, negate)
        if tok and tok[0] == "op" and tok[1] in _CMP:
            self.i += 1
            return ("cmp", tok[1], left, self.sum_())
        return left

    def sum_(self):
        node = self.term()
        while (tok := self.peek()) and tok[0] == "op" and tok[1] in "+-":
            self.i += 1
            node = ("arith", tok[1], node, self.term())
        return node

    def term(self):
        node = self.atom()
        while (tok := self.peek()) and tok[0] == "op" and tok[1] in "*/":
            self.i += 1
            node = ("arith", tok[1], node, self.atom())
        return node

    def atom(self):
        tok = self.peek()
        if tok is None:
            raise ConditionError(f"условие «{self.text}» обрывается")
        kind, value = tok
        self.i += 1
        if kind == "num":
            return ("num", float(value.replace(",", ".")))
        if kind == "param":
            return ("param", value)
        if kind == "exc":
            return ("exc", value)
        if kind == "lp":
            node = self.or_()
            self.take("rp")
            return node
        if kind == "word":
            if self.accept("lp"):
                args = []
                if not self.accept("rp"):
                    args.append(self.call_arg())
                    while self.accept("comma"):
                        args.append(self.call_arg())
                    self.take("rp")
                return ("call", value, args)
            return ("var", value)
        raise ConditionError(f"в условии «{self.text}» неожиданное «{value}»")

    def call_arg(self):
        if self.accept("excany"):
            return ("excany",)
        return self.or_()


# ---------------------------------------------------------------------------
# Вычисление
# ---------------------------------------------------------------------------


@dataclass
class Env:
    """Данные для вычисления условия на одном месяце.

    fields  — поле → массив (числа float с NaN, флаги bool, даты datetime64) или одно значение;
    results — предмет «результата» (info, notice, restriction) → массив bool;
    exceptions — EX-xx → массив bool (сейчас исключения не учитываются: всё «Нет»);
    today   — дата, от которой считаются days_since / months_since (01.M);
    size    — число ЛС (длина массивов).
    """

    fields: dict[str, Any]
    results: dict[str, Any]
    today: np.datetime64
    size: int
    exceptions: dict[str, Any] | None = None


def _as_bool(value, size: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype != bool:
        arr = np.asarray(pd.array(arr, dtype="boolean").fillna(False), dtype=bool) if arr.ndim else np.asarray(bool(arr))
    return np.broadcast_to(arr, (size,))


def _is_null(value) -> np.ndarray:
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.datetime64):
        return np.isnat(arr)
    if np.issubdtype(arr.dtype, np.floating):
        return np.isnan(arr)
    return np.asarray(pd.isna(arr), dtype=bool)


def _month_index(dates) -> np.ndarray:
    d = np.asarray(dates, dtype="datetime64[M]")
    out = d.astype("int64").astype(float)
    out[np.isnat(d)] = np.nan
    return out


class Condition:
    """Разобранное условие. evaluate(env) → массив bool длины env.size."""

    def __init__(self, text: str, params: dict[str, float | None], where: str = ""):
        self.text = text
        self.where = where
        self.params = params
        self.tree = _Parser(text).parse()
        self._check(self.tree)

    def _fail(self, msg: str) -> ConditionError:
        prefix = f"{self.where}: " if self.where else ""
        return ConditionError(f"{prefix}{msg} (условие «{self.text}»)")

    def _check(self, node) -> None:
        kind = node[0]
        if kind == "param":
            code = node[1]
            if code not in self.params:
                raise self._fail(f"нет параметра {code}")
            if self.params[code] is None:
                raise self._fail(f"параметр {code} не число, использовать его в условии нельзя")
        for child in node[1:]:
            if isinstance(child, tuple):
                self._check(child)
            elif isinstance(child, list):
                for c in child:
                    self._check(c)

    def evaluate(self, env: Env) -> np.ndarray:
        return _as_bool(self._eval(self.tree, env), env.size)

    # -- узлы --

    def _eval(self, node, env: Env):
        kind = node[0]
        if kind == "num":
            return node[1]
        if kind == "param":
            return self.params[node[1]]
        if kind == "var":
            name = node[1]
            if name not in env.fields:
                raise self._fail(f"нет данных для поля «{name}»")
            return env.fields[name]
        if kind == "exc":
            return (env.exceptions or {}).get(node[1], False)
        if kind == "and":
            return np.logical_and(_as_bool(self._eval(node[1], env), env.size),
                                  _as_bool(self._eval(node[2], env), env.size))
        if kind == "or":
            return np.logical_or(_as_bool(self._eval(node[1], env), env.size),
                                 _as_bool(self._eval(node[2], env), env.size))
        if kind == "not":
            return np.logical_not(_as_bool(self._eval(node[1], env), env.size))
        if kind == "isnull":
            res = _is_null(self._eval(node[1], env))
            return ~res if node[2] else res
        if kind == "cmp":
            left, right = self._eval(node[2], env), self._eval(node[3], env)
            with np.errstate(invalid="ignore"):
                return _CMP[node[1]](left, right)
        if kind == "arith":
            left, right = self._eval(node[2], env), self._eval(node[3], env)
            with np.errstate(invalid="ignore", divide="ignore"):
                return _ARITH[node[1]](np.asarray(left, dtype=float), np.asarray(right, dtype=float))
        if kind == "call":
            return self._call(node[1], node[2], env)
        raise self._fail(f"неожиданный элемент «{kind}»")

    def _date_arg(self, arg, env: Env):
        """Аргумент days_since / months_since: поле-дата или мера (claim → last_claim_date)."""
        if arg[0] == "var" and arg[1] not in env.fields:
            for alias in (f"{arg[1]}_date", f"last_{arg[1]}_date"):
                if alias in env.fields:
                    return env.fields[alias]
        return self._eval(arg, env)

    def _call(self, name: str, args: list, env: Env):
        if name == "result":
            if len(args) != 1 or args[0][0] != "var":
                raise self._fail("result(…) ожидает один предмет: info, notice, restriction")
            subject = args[0][1]
            if subject not in env.results:
                raise self._fail(f"нет данных о результате «{subject}»")
            return env.results[subject]
        if name == "days_since":
            dates = np.asarray(self._date_arg(args[0], env), dtype="datetime64[D]")
            days = (np.datetime64(env.today, "D") - dates).astype("int64").astype(float)
            days[np.isnat(dates)] = np.nan
            return days
        if name == "months_since":
            return _month_index(np.broadcast_to(np.datetime64(env.today, "M"), (env.size,))) - _month_index(
                self._date_arg(args[0], env))
        if name == "any":
            if args and args[0][0] == "excany":
                ex = env.exceptions or {}
                out = np.zeros(env.size, dtype=bool)
                for v in ex.values():
                    out |= _as_bool(v, env.size)
                return out
        raise self._fail(f"функция «{name}» пока не поддерживается")


def compile_condition(text: str, params: dict[str, float | None], where: str = "") -> Condition:
    return Condition(text, params, where)


def params_of(book) -> dict[str, float | None]:
    """Числовые значения параметров книги: {"P-01": 500.0, "P-02": None, …}."""
    return {code: p.value_num for code, p in book.params.items()}
