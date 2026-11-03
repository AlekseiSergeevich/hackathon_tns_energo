"""Проверка входных файлов 01–14: что считается ошибкой, что предупреждением и как об этом сказать.

Загрузчики (tns/data/loaders.py) записывают найденные проблемы в список Problem. Ошибка
останавливает запуск, предупреждение — нет. load_all собирает ошибки по всем файлам и выдаёт
их одним исключением ValidationError, чтобы пользователь исправил всё за один раз.

Потери при разборе (номер ЛС, дата, сумма не прочитались, строка убрана) — предупреждение.
Оно становится ошибкой, если не прочитано ни одного значения или доля потерь больше
validation.max_lost_share из configs/settings.yaml.
"""
from __future__ import annotations

import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

ERROR, WARNING = "error", "warning"
MAX_EXAMPLES = 3


@dataclass(frozen=True)
class Problem:
    level: str  # ERROR или WARNING
    file: str  # имя входного файла, пусто — проблема не про один файл
    message: str  # законченная фраза для пользователя
    column: str = ""
    count: int = 0  # для потерь при разборе: сколько значений не прочитано
    total: int = 0  # из скольких
    expected: str = ""  # что ожидалось в колонке, например «дата ДД.ММ.ГГГГ»
    blocking: bool = True  # False — потеря никогда не становится ошибкой (необязательная колонка)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Problem":
        return cls(**data)


class ValidationError(ValueError):
    """Входные файлы не прошли проверку. Все проблемы — в .problems."""

    def __init__(self, problems: list[Problem], data_dir: Path | str | None = None):
        self.problems = list(problems)
        super().__init__(format_report(self.problems, data_dir))

    @property
    def errors(self) -> list[Problem]:
        return [p for p in self.problems if p.level == ERROR]


def error(file: str, message: str, column: str = "") -> Problem:
    return Problem(ERROR, file, message, column)


def warning(file: str, message: str, column: str = "") -> Problem:
    return Problem(WARNING, file, message, column)


# ---------------------------------------------------------------------------
# Числа и слова
# ---------------------------------------------------------------------------

def plural(n: int, one: str, few: str, many: str) -> str:
    """plural(3, "ошибка", "ошибки", "ошибок") → «ошибки»."""
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def number(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def percent(share: float) -> str:
    if 0 < share < 0.0005:
        return "менее 0,1 %"
    return f"{share * 100:.1f} %".replace(".", ",")


def repeats(n: int) -> str:
    return f"{number(n)} {plural(n, 'повтор', 'повтора', 'повторов')}"


def _shown(value) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "пусто"
    text = str(value).strip()
    return f"«{text[:30]}…»" if len(text) > 30 else f"«{text}»"


# ---------------------------------------------------------------------------
# Типовые проблемы
# ---------------------------------------------------------------------------

def lost_values(
    file: str, column: str, bad, values, rows, *, what: str,
    blocking: bool = True, action: str = "такие строки убраны",
) -> Problem | None:
    """Предупреждение о значениях, которые не прочитались. None — если таких нет.

    bad — маска непрочитанных значений, values — исходные значения (для примеров),
    rows — номера строк в файле (как их видит пользователь в Excel или блокноте); все три одной длины.
    """
    bad = np.asarray(bad, dtype=bool)
    count, total = int(bad.sum()), len(bad)
    if not count:
        return None
    values, rows = np.asarray(values, dtype=object), np.asarray(rows)
    positions = np.flatnonzero(bad)[:MAX_EXAMPLES]
    examples = ", ".join(f"строка {number(int(rows[p]))} — {_shown(values[p])}" for p in positions)
    message = (
        f"В файле «{file}» в колонке «{column}» не прочитаны {number(count)} из {number(total)} значений "
        f"({percent(count / total)}): ожидается {what}; {action}. Например: {examples}."
    )
    return Problem(WARNING, file, message, column=column, count=count, total=total, expected=what, blocking=blocking)


def escalate(problems: list[Problem], max_lost_share: float) -> list[Problem]:
    """Превращает слишком большие потери при разборе в ошибки."""
    result = []
    for problem in problems:
        if problem.level == WARNING and problem.blocking and problem.total:
            if problem.count == problem.total:
                problem = replace(problem, level=ERROR, message=(
                    f"В файле «{problem.file}» в колонке «{problem.column}» не прочитано ни одного значения "
                    f"из {number(problem.total)}: ожидается {problem.expected}. "
                    "Проверьте, что в колонке нужные данные и колонки не сдвинуты."
                ))
            elif problem.count / problem.total > max_lost_share:
                problem = replace(problem, level=ERROR, message=(
                    problem.message + f" Это больше допустимого ({percent(max_lost_share)}, настройка "
                    "validation.max_lost_share): похоже, файл испорчен или колонка заполнена не тем."
                ))
        result.append(problem)
    return result


def open_error(file: str, error: BaseException, kind: str) -> str:
    """Понятное сообщение, если файл не удалось открыть. kind — «excel» или «csv»."""
    if isinstance(error, PermissionError):
        return f"Файл «{file}» не открывается: нет доступа. Если он открыт в Excel, закройте его и запустите снова."
    if isinstance(error, IsADirectoryError):
        return f"«{file}» — это папка, а нужен файл."
    if kind == "csv":
        if isinstance(error, pd.errors.EmptyDataError):
            return f"Файл «{file}» пуст. Нужна шапка «Номер;Дата оплаты;Сумма;Способ оплаты» и строки оплат."
        if isinstance(error, UnicodeDecodeError):
            return (
                f"Файл «{file}» не читается в кодировке UTF-8. Пересохраните его как «CSV UTF-8 "
                "(разделитель — точка с запятой)»."
            )
        return f"Файл «{file}» не читается как CSV с разделителем «;»: {error}."
    if isinstance(error, (zipfile.BadZipFile, KeyError)) or "file format cannot be determined" in str(error):
        return (
            f"Файл «{file}» не открывается как книга Excel: похоже, он повреждён или сохранён в другом формате. "
            "Откройте его в Excel и сохраните как «Книга Excel (*.xlsx)»."
        )
    return f"Файл «{file}» не открывается как книга Excel: {error}."


# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------

def format_report(problems: list[Problem], data_dir: Path | str | None = None) -> str:
    """Текст для пользователя: все ошибки списком и число предупреждений."""
    errors = [p for p in problems if p.level == ERROR]
    warnings = [p for p in problems if p.level == WARNING]
    lines = [
        f"Входные файлы не прошли проверку: {number(len(errors))} {plural(len(errors), 'ошибка', 'ошибки', 'ошибок')}."
    ]
    lines += [f"  {i}. {p.message}" for i, p in enumerate(errors, start=1)]
    if warnings:
        lines.append(
            f"Кроме того, {number(len(warnings))} {plural(len(warnings), 'предупреждение', 'предупреждения', 'предупреждений')} "
            "(они запуск не останавливают, см. сообщения выше)."
        )
    where = f" в папке «{data_dir}»" if data_dir is not None else ""
    lines.append(f"Исправьте файлы{where} и запустите снова.")
    return "\n".join(lines)
