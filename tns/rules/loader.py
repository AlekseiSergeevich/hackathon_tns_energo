"""
Загрузчик таблицы правил воздействия.
Читает книгу Excel «Правила воздействия ТНС vX.Y.xlsx» и превращает её в объект
RuleBook. 
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import openpyxl

logger = logging.getLogger(__name__)

# Справочники

# Названия мер в книге правил (листы «Правила» и «Лимиты») → единые ключи мер
# из docs/interfaces.md. Сравнение без учёта регистра и лишних пробелов,
# номер файла в начале («04 Автодозвон») отбрасывается.
MEASURE_KEYS: dict[str, str] = {
    "автодозвон": "autodial",
    "e-mail": "email",
    "смс": "sms",
    "обзвон оператором": "operator_call",
    "претензия": "claim",
    "выезд к абоненту": "visit",
    "уведомление о введении ограничения": "restriction_notice",
    "ограничение": "restriction",
    "ограничение подачи электроэнергии": "restriction",
    "заявление о выдаче судебного приказа": "court_application",
    "получение судебного приказа или ил": "court_result",
    "получение судебного приказа / ил": "court_result",
    "список на комиссию (не заявление)": "commission_list",
    "информация в квитанции": "receipt_info",
}

CONDITION_SUBJECTS = {"info", "notice", "restriction", "claim"}

SHEETS_TO_SKIP = {"Как читать"}

REQUIRED_COLUMNS: dict[str, dict[str, str]] = {
    "Параметры": {
        "code": "Код",
        "name": "Параметр",
        "value": "Значение",
        "unit": "Ед.",
    },
    "Стадии": {
        "code": "Код",
        "name": "Стадия",
        "meaning": "Смысл",
        "allowed": "Разрешённые меры",
    },
    "Переходы": {
        "code": "Код",
        "from": "Из",
        "to": "В",
        "human": "Условие (по-человечески)",
        "formal": "Условие (формально, поля из «Поля данных»)",
    },
    "Исключения": {
        "code": "Код",
        "name": "Исключение",
        "blocks": "Что блокирует",
        "fields": "Поле",
    },
    "Правила": {
        "code": "Код",
        "procedure": "Процедура",
        "measure": "Мера",
        "stages": "Стадии",
        "overdue_from": "Просрочка от, мес.",
        "overdue_to": "Просрочка до, мес.",
        "dz_from": "ДЗ от, руб.",
        "dz_to": "ДЗ до, руб.",
        "human": "Доп. условие (по-человечески)",
        "formal": "Доп. условие (формально)",
        "background": "Фоновая?",
        "active": "Активно",
        "template": "Шаблон объяснения",
    },
    "Лимиты": {
        "measure": "Мера",
        "rule_codes": "Код правил",
        "limit": "Лимит в месяц",
    },
    "Поля данных": {
        "field": "Поле",
        "meaning": "Смысл",
        "available": "Данные хакатона",
        "how": "Как считаем / допущение",
    },
    "Примеры": {
        "number": "№",
        "situation": "Ситуация (вход)",
        "prev_stage": "Стадия прошлого месяца",
        "transition": "Переход",
        "stage": "Стадия сейчас",
        "candidates": "Кандидаты (сработавшие правила)",
        "refused": "Не назначено (причина)",
    },
}

OPTIONAL_COLUMNS: dict[str, dict[str, str]] = {
    "Лимиты": {"note": "Примечание"},
}

CODE_PATTERNS = {
    "Параметры": re.compile(r"^P-\d{2}$"),
    "Стадии": re.compile(r"^S(\d|X)$"),
    "Переходы": re.compile(r"^T-\S+→\S+$"),
    "Исключения": re.compile(r"^EX-\d{2}$"),
    "Правила": re.compile(r"^R-[A-Z]{3}-\d{2}$"),
}
CODE_EXAMPLES = {
    "Параметры": "P-01",
    "Стадии": "S1 или SX",
    "Переходы": "T-1→2",
    "Исключения": "EX-01",
    "Правила": "R-SMS-01",
}

EMPTY_MARKS = {"", "—", "-", "–"}
SAME_AS_ABOVE = {"то же", "то же."}
NO_LIMIT_MARKS = {"нет ограничений", "без ограничений"}
YES = {"да"}
NO = {"нет"}

RE_PARAM_REF = re.compile(r"\bP-\d{2}\b")
RE_EXC_REF = re.compile(r"\bEX-\d{2}\b")
RE_IDENT = re.compile(r"(?<![\w-])([a-z_][a-z0-9_]*)(\s*\()?")
RE_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
RE_RANGE_CODES = re.compile(r"^(R-[A-Z]{3}-)(\d{2})\.\.(\d{2})$")
RE_MASK_CODES = re.compile(r"^(R-[A-Z]{3}-)\*$")
RE_VERSION = re.compile(r"v(\d+(?:\.\d+)*)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Структуры данных
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Param:
    code: str
    name: str
    value_raw: Any  # число или текст
    value_num: float | None  # число, если значение числовое; для текстовых — None
    unit: str | None


@dataclass(frozen=True)
class Stage:
    code: str
    name: str
    meaning: str | None
    allowed_measures_text: str | None  # текст колонки «Разрешённые меры» (справочно)


@dataclass(frozen=True)
class Transition:
    code: str
    from_stages: tuple[str, ...]
    to_stage: str | None  # None — возврат на стадию до исключения (T-X→пред.)
    returns_to_previous: bool
    condition_text: str | None
    condition: str | None  # формальное условие (псевдокод, исполняет машина стадий)
    order: int  # порядковый номер строки на листе


@dataclass(frozen=True)
class ExceptionRule:
    code: str
    name: str
    blocks_text: str | None
    fields: tuple[str, ...]


@dataclass(frozen=True)
class Rule:
    code: str
    procedure: str | None
    measure: str  # ключ меры: sms, visit, …
    measure_name: str  # как написано в таблице
    stages: tuple[str, ...]
    overdue_from: float | None
    overdue_to: float | None
    dz_from: float | None
    dz_to: float | None
    condition_text: str | None
    condition: str | None  # формальное условие; «то же» уже раскрыто
    is_background: bool
    is_active: bool
    template: str | None
    placeholders: tuple[str, ...]  # {dz}, {overdue_m}, … из шаблона


@dataclass(frozen=True)
class Limit:
    measure: str
    measure_name: str
    value: int | None  # None — нет ограничений
    rule_codes: tuple[str, ...]
    note: str | None


@dataclass(frozen=True)
class DataField:
    names: tuple[str, ...]  # «call_failed / refused» → два имени
    meaning: str | None
    available: str | None
    how: str | None


@dataclass(frozen=True)
class Example:
    number: int
    situation: str
    prev_stage: str | None
    transition: str | None
    stage: str | None
    candidates: str | None
    refused: str | None


@dataclass(frozen=True)
class RuleBookVersion:
    file_name: str
    label: str | None  # «0.3» из имени файла
    sha256: str
    modified: datetime

    def __str__(self) -> str:
        label = f"v{self.label}" if self.label else "версия не указана в имени файла"
        return (
            f"{self.file_name} ({label}), sha256 {self.sha256[:12]}, "
            f"изменён {self.modified:%d.%m.%Y %H:%M}"
        )


@dataclass
class RuleBook:
    version: RuleBookVersion
    params: dict[str, Param]
    stages: dict[str, Stage]
    transitions: list[Transition]
    exceptions: dict[str, ExceptionRule]
    rules: dict[str, Rule]
    limits: dict[str, Limit]
    fields: list[DataField]
    examples: list[Example]
    warnings: list[str] = field(default_factory=list)

    @property
    def active_rules(self) -> dict[str, Rule]:
        return {code: r for code, r in self.rules.items() if r.is_active}

    @property
    def field_names(self) -> set[str]:
        return {name for f in self.fields for name in f.names}

    def rules_for_measure(self, measure: str, active_only: bool = True) -> list[Rule]:
        rules = self.active_rules if active_only else self.rules
        return [r for r in rules.values() if r.measure == measure]


class RuleBookError(ValueError):
    """Таблица правил содержит ошибки. В .errors — список сообщений."""

    def __init__(self, path: Path, errors: list[str]):
        self.path = path
        self.errors = errors
        lines = "\n".join(f"  • {e}" for e in errors)
        super().__init__(
            f"Таблица правил «{path.name}» содержит ошибки ({len(errors)}). "
            f"Исправьте их в Excel и запустите снова:\n{lines}"
        )

# Чтение листов

@dataclass
class _Row:
    sheet: str
    number: int  # номер строки в Excel
    values: dict[str, Any]  # внутреннее имя колонки → значение

    def where(self, key: str | None = None) -> str:
        if key is None:
            return f"лист «{self.sheet}», строка {self.number}"
        header = REQUIRED_COLUMNS[self.sheet][key]
        return f"лист «{self.sheet}», строка {self.number}, колонка «{header}»"


class _Collector:
    """Копит ошибки и предупреждения."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def _clean(value: Any) -> Any:
    if isinstance(value, str):
        value = re.sub(r"\s+", " ", value).strip()
        return value or None
    return value


def _read_sheet(wb: openpyxl.Workbook, sheet: str, col: _Collector) -> list[_Row] | None:
    """Читает лист: первая строка — заголовки; строки-сноски пропускаются.
    """
    if sheet not in wb.sheetnames:
        col.error(f"нет листа «{sheet}»")
        return None
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        col.error(f"лист «{sheet}» пустой")
        return None

    headers = [_clean(h) for h in rows[0]]
    positions: dict[str, int] = {}
    missing = []
    for key, title in REQUIRED_COLUMNS[sheet].items():
        if title in headers:
            positions[key] = headers.index(title)
        else:
            missing.append(title)
    if missing:
        names = ", ".join(f"«{m}»" for m in missing)
        col.error(f"лист «{sheet}»: нет обязательных колонок {names} (заголовки — в первой строке)")
        return None
    for key, title in OPTIONAL_COLUMNS.get(sheet, {}).items():
        if title in headers:
            positions[key] = headers.index(title)

    result = []
    for i, raw in enumerate(rows[1:], start=2):
        cells = [_clean(v) for v in raw]
        if sum(v is not None for v in cells) <= 1:
            continue  # пустая строка или сноска
        values = {key: cells[pos] if pos < len(cells) else None for key, pos in positions.items()}
        for key in OPTIONAL_COLUMNS.get(sheet, {}):
            values.setdefault(key, None)
        result.append(_Row(sheet, i, values))
    return result


# Разбор значений

def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value in EMPTY_MARKS)


def _to_number(value: Any) -> float | None:
    """Число из ячейки. Пусто → None. Нечисловой текст → ValueError."""
    if _is_empty(value):
        return None
    if isinstance(value, bool):
        raise ValueError(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(" ", "").replace(" ", "").replace(",", ".")
    return float(text)


def _number_cell(row: _Row, key: str, col: _Collector) -> float | None:
    value = row.values[key]
    try:
        number = _to_number(value)
    except (ValueError, TypeError):
        col.error(f"{row.where(key)}: ожидается число или пустая ячейка, а записано «{value}»")
        return None
    if number is not None and number < 0:
        col.error(f"{row.where(key)}: отрицательное значение {value}")
    return number


def _bool_cell(row: _Row, key: str, col: _Collector) -> bool:
    value = row.values[key]
    text = str(value).strip().lower() if value is not None else ""
    if text in YES:
        return True
    if text in NO:
        return False
    col.error(f"{row.where(key)}: ожидается «Да» или «Нет», а записано «{value or 'пусто'}»")
    return False


def _measure_key(name: Any) -> str | None:
    if name is None:
        return None
    text = re.sub(r"^\d{2}\s+", "", str(name)).strip().lower()
    return MEASURE_KEYS.get(text)


def _parse_stage_list(text: Any, known: list[str], allow_any: bool) -> tuple[str, ...]:
    """«S1–S4» → (S1, S2, S3, S4); «S2, S3» → (S2, S3); «любая» → все стадии.

    Неизвестная стадия → ValueError с понятным текстом.
    """
    if _is_empty(text):
        raise ValueError("стадия не указана")
    text = str(text).strip()
    if allow_any and text.lower() in {"любая", "любые", "все"}:
        return tuple(known)
    result: list[str] = []
    for part in re.split(r"\s*[,;]\s*", text):
        m = re.fullmatch(r"(S\w)\s*[–—-]\s*(S\w)", part)
        if m:
            a, b = m.groups()
            for s in (a, b):
                if s not in known:
                    raise ValueError(f"стадия «{s}» не найдена на листе «Стадии»")
            ia, ib = known.index(a), known.index(b)
            if ia > ib:
                raise ValueError(f"диапазон «{part}» задан в обратном порядке")
            result.extend(known[ia : ib + 1])
        else:
            if part not in known:
                raise ValueError(f"стадия «{part}» не найдена на листе «Стадии»")
            result.append(part)
    return tuple(dict.fromkeys(result))


def _condition_identifiers(condition: str) -> set[str]:
    """Имена полей из формального условия (без функций и предметов результата)."""
    names = set()
    for m in RE_IDENT.finditer(condition):
        name, is_call = m.group(1), m.group(2)
        if is_call or name in CONDITION_SUBJECTS:
            continue
        names.add(name)
    return names


def _field_known(name: str, field_names: set[str]) -> bool:
    if name in field_names:
        return True
    # маски вида «*_date (по мерам)»
    for pattern in field_names:
        if "*" in pattern and re.fullmatch(pattern.replace("*", r"\w+"), name):
            return True
    return False


# Разбор листов


def _check_code(row: _Row, col: _Collector, seen: dict[str, int]) -> str | None:
    code = row.values["code"]
    if code is None:
        col.error(f"{row.where('code')}: не заполнен код (строка с данными без кода)")
        return None
    code = str(code)
    pattern = CODE_PATTERNS[row.sheet]
    if not pattern.match(code):
        col.error(
            f"{row.where('code')}: код «{code}» не по шаблону "
            f"(пример: {CODE_EXAMPLES[row.sheet]})"
        )
        return None
    if code in seen:
        col.error(f"{row.where('code')}: код {code} уже встречался в строке {seen[code]}")
        return None
    seen[code] = row.number
    return code


def _parse_params(rows: list[_Row], col: _Collector) -> dict[str, Param]:
    params: dict[str, Param] = {}
    seen: dict[str, int] = {}
    for row in rows:
        code = _check_code(row, col, seen)
        if code is None:
            continue
        raw = row.values["value"]
        if _is_empty(raw):
            col.error(f"{row.where('value')}: у параметра {code} нет значения")
        try:
            num = _to_number(raw)
        except (ValueError, TypeError):
            num = None  # текстовый параметр (P-02, P-16, …)
        params[code] = Param(code, row.values["name"] or "", raw, num, row.values["unit"])
    return params


def _parse_stages(rows: list[_Row], col: _Collector) -> dict[str, Stage]:
    stages: dict[str, Stage] = {}
    seen: dict[str, int] = {}
    for row in rows:
        code = _check_code(row, col, seen)
        if code is None:
            continue
        stages[code] = Stage(code, row.values["name"] or "", row.values["meaning"], row.values["allowed"])
    return stages


def _parse_transitions(rows: list[_Row], known: list[str], col: _Collector) -> list[Transition]:
    result = []
    seen: dict[str, int] = {}
    for order, row in enumerate(rows, start=1):
        code = _check_code(row, col, seen)
        if code is None:
            continue
        try:
            from_stages = _parse_stage_list(row.values["from"], known, allow_any=True)
        except ValueError as e:
            col.error(f"{row.where('from')}: {e}")
            from_stages = ()
        to_raw = row.values["to"]
        to_stage, returns = None, False
        if to_raw is not None and "до исключения" in str(to_raw).lower():
            returns = True
        elif to_raw in known:
            to_stage = to_raw
        else:
            col.error(
                f"{row.where('to')}: стадия «{to_raw or 'пусто'}» не найдена на листе «Стадии» "
                f"(допустимо также «стадия до исключения»)"
            )
        formal = None if _is_empty(row.values["formal"]) else row.values["formal"]
        result.append(
            Transition(code, from_stages, to_stage, returns, row.values["human"], formal, order)
        )
    return result


def _parse_exceptions(rows: list[_Row], col: _Collector) -> dict[str, ExceptionRule]:
    result: dict[str, ExceptionRule] = {}
    seen: dict[str, int] = {}
    for row in rows:
        code = _check_code(row, col, seen)
        if code is None:
            continue
        raw = row.values["fields"]
        fields = tuple(f for f in re.split(r"\s*[,/;]\s*", raw or "") if f)
        result[code] = ExceptionRule(code, row.values["name"] or "", row.values["blocks"], fields)
    return result


def _parse_rules(rows: list[_Row], known: list[str], col: _Collector) -> dict[str, Rule]:
    rules: dict[str, Rule] = {}
    seen: dict[str, int] = {}
    prev_formal: str | None = None
    prev_human: str | None = None
    for row in rows:
        code = _check_code(row, col, seen)
        v = row.values

        measure_name = v["measure"]
        measure = _measure_key(measure_name)
        if measure is None:
            col.error(
                f"{row.where('measure')}: неизвестная мера «{measure_name or 'пусто'}». "
                f"Допустимые: {', '.join(sorted({k for k in MEASURE_KEYS}))}"
            )

        try:
            stages = _parse_stage_list(v["stages"], known, allow_any=False)
        except ValueError as e:
            col.error(f"{row.where('stages')}: {e}")
            stages = ()

        nums = {k: _number_cell(row, k, col) for k in ("overdue_from", "overdue_to", "dz_from", "dz_to")}
        for lo, hi, title in (("overdue_from", "overdue_to", "просрочка"), ("dz_from", "dz_to", "ДЗ")):
            if nums[lo] is not None and nums[hi] is not None and nums[lo] > nums[hi]:
                col.error(
                    f"{row.where(lo)}: {title} «от» ({nums[lo]:g}) больше, чем «до» ({nums[hi]:g})"
                )

        background = _bool_cell(row, "background", col)
        active = _bool_cell(row, "active", col)

        formal = v["formal"]
        if isinstance(formal, str) and formal.lower() in SAME_AS_ABOVE:
            if prev_formal is None:
                col.error(f"{row.where('formal')}: «то же», но выше нет условия, на которое можно сослаться")
            formal = prev_formal
        elif _is_empty(formal):
            formal = None
        human = v["human"]
        if isinstance(human, str) and human.lower() in SAME_AS_ABOVE:
            human = prev_human
        prev_formal, prev_human = formal, human

        template = None if _is_empty(v["template"]) else v["template"]
        placeholders = tuple(dict.fromkeys(RE_PLACEHOLDER.findall(template or "")))
        if active and template is None:
            col.error(f"{row.where('template')}: у активного правила {code} нет шаблона объяснения")

        if code is None or measure is None:
            continue
        rules[code] = Rule(
            code=code,
            procedure=v["procedure"],
            measure=measure,
            measure_name=str(measure_name),
            stages=stages,
            overdue_from=nums["overdue_from"],
            overdue_to=nums["overdue_to"],
            dz_from=nums["dz_from"],
            dz_to=nums["dz_to"],
            condition_text=human,
            condition=formal,
            is_background=background,
            is_active=active,
            template=template,
            placeholders=placeholders,
        )
    return rules


def _expand_rule_codes(text: str, all_codes: list[str]) -> list[str]:
    """«R-ADL-*» → все правила с префиксом; «R-CRT-01..03» → диапазон; «A, B» → список."""
    result: list[str] = []
    for part in re.split(r"\s*[,;]\s*", text):
        if not part:
            continue
        m_mask = RE_MASK_CODES.match(part)
        m_range = RE_RANGE_CODES.match(part)
        if m_mask:
            result.extend(c for c in all_codes if c.startswith(m_mask.group(1)))
        elif m_range:
            prefix, a, b = m_range.group(1), int(m_range.group(2)), int(m_range.group(3))
            result.extend(f"{prefix}{i:02d}" for i in range(a, b + 1))
        else:
            result.append(part)
    return list(dict.fromkeys(result))


def _parse_limits(rows: list[_Row], rules: dict[str, Rule], col: _Collector) -> dict[str, Limit]:
    limits: dict[str, Limit] = {}
    seen: dict[str, int] = {}
    all_codes = list(rules)
    for row in rows:
        v = row.values
        measure = _measure_key(v["measure"])
        if measure is None:
            col.error(f"{row.where('measure')}: неизвестная мера «{v['measure'] or 'пусто'}»")
            continue
        if measure in seen:
            col.error(f"{row.where('measure')}: мера «{v['measure']}» уже встречалась в строке {seen[measure]}")
            continue
        seen[measure] = row.number

        raw = v["limit"]
        value: int | None = None
        if isinstance(raw, str) and raw.lower() in NO_LIMIT_MARKS:
            value = None
        else:
            try:
                num = _to_number(raw)
            except (ValueError, TypeError):
                num = None
            if num is None or num < 0 or num != int(num):
                col.error(
                    f"{row.where('limit')}: ожидается целое число или «нет ограничений», "
                    f"а записано «{raw or 'пусто'}»"
                )
            else:
                value = int(num)

        codes_text = v["rule_codes"] or ""
        codes = _expand_rule_codes(codes_text, all_codes)
        if not codes:
            col.error(f"{row.where('rule_codes')}: «{codes_text or 'пусто'}» не соответствует ни одному правилу")
        for c in codes:
            if c not in rules:
                col.error(f"{row.where('rule_codes')}: правила {c} нет на листе «Правила»")
            elif rules[c].measure != measure:
                col.error(
                    f"{row.where('rule_codes')}: правило {c} относится к мере "
                    f"«{rules[c].measure_name}», а не к «{v['measure']}»"
                )
        limits[measure] = Limit(measure, str(v["measure"]), value, tuple(codes), v.get("note"))
    return limits


def _parse_fields(rows: list[_Row]) -> list[DataField]:
    result = []
    for row in rows:
        raw = row.values["field"]
        if raw is None:
            continue
        text = re.sub(r"\(.*?\)", "", str(raw))  # «*_date (по мерам)» → «*_date»
        names = tuple(n for n in re.split(r"\s*[,/;]\s*", text.strip()) if n)
        result.append(DataField(names, row.values["meaning"], row.values["available"], row.values["how"]))
    return result


def _parse_examples(rows: list[_Row], col: _Collector) -> list[Example]:
    result = []
    for row in rows:
        v = row.values
        try:
            number = int(_to_number(v["number"]) or 0)
        except (ValueError, TypeError):
            col.error(f"{row.where('number')}: номер примера должен быть числом, а записано «{v['number']}»")
            continue
        result.append(
            Example(number, v["situation"] or "", v["prev_stage"], v["transition"], v["stage"],
                    v["candidates"], v["refused"])
        )
    return result


# Перекрёстные проверки


def _check_references(
    params: dict[str, Param],
    exceptions: dict[str, ExceptionRule],
    transitions: list[Transition],
    rules: dict[str, Rule],
    field_names: set[str],
    col: _Collector,
) -> None:
    sources: list[tuple[str, str | None]] = []
    sources += [(f"переход {t.code}", t.condition) for t in transitions]
    sources += [(f"правило {r.code}", r.condition) for r in rules.values()]

    unknown_fields: dict[str, list[str]] = {}
    for where, cond in sources:
        if not cond:
            continue
        for ref in RE_PARAM_REF.findall(cond):
            if ref not in params:
                col.error(f"{where} ссылается на параметр {ref}, которого нет на листе «Параметры»")
        for ref in RE_EXC_REF.findall(cond):
            if ref not in exceptions:
                col.error(f"{where} ссылается на исключение {ref}, которого нет на листе «Исключения»")
        for name in sorted(_condition_identifiers(cond)):
            if not _field_known(name, field_names):
                unknown_fields.setdefault(name, []).append(where)

    for exc in exceptions.values():
        for name in exc.fields:
            if not _field_known(name, field_names):
                unknown_fields.setdefault(name, []).append(f"исключение {exc.code}")

    for name, places in unknown_fields.items():
        col.warn(f"поле «{name}» не описано на листе «Поля данных» (используется: {', '.join(places)})")


# Главная функция

def _version(path: Path) -> RuleBookVersion:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    m = RE_VERSION.search(path.stem)
    return RuleBookVersion(
        file_name=path.name,
        label=m.group(1) if m else None,
        sha256=digest,
        modified=datetime.fromtimestamp(path.stat().st_mtime),
    )


def load_rulebook(path: str | Path) -> RuleBook:
    """Читает и проверяет таблицу правил. При ошибках — RuleBookError со списком."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Таблица правил не найдена: {path}")
    try:
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    except Exception as e:  # повреждённый файл, не Excel, открыт с паролем
        raise RuleBookError(path, [f"файл не удалось открыть как Excel (.xlsx): {e}"]) from e

    col = _Collector()
    try:
        sheets = {name: _read_sheet(wb, name, col) for name in REQUIRED_COLUMNS}
    finally:
        wb.close()

    params = _parse_params(sheets["Параметры"] or [], col)
    stages = _parse_stages(sheets["Стадии"] or [], col)
    known_stages = list(stages)
    transitions = _parse_transitions(sheets["Переходы"] or [], known_stages, col)
    exceptions = _parse_exceptions(sheets["Исключения"] or [], col)
    rules = _parse_rules(sheets["Правила"] or [], known_stages, col)
    limits = _parse_limits(sheets["Лимиты"] or [], rules, col)
    fields = _parse_fields(sheets["Поля данных"] or [])
    examples = _parse_examples(sheets["Примеры"] or [], col)

    field_names = {n for f in fields for n in f.names}
    _check_references(params, exceptions, transitions, rules, field_names, col)

    if col.errors:
        raise RuleBookError(path, col.errors)

    book = RuleBook(
        version=_version(path),
        params=params,
        stages=stages,
        transitions=transitions,
        exceptions=exceptions,
        rules=rules,
        limits=limits,
        fields=fields,
        examples=examples,
        warnings=col.warnings,
    )
    logger.info("Таблица правил: %s", book.version)
    for w in book.warnings:
        logger.warning("Таблица правил: %s", w)
    return book


# Консоль

def summary(book: RuleBook) -> str:
    inactive = [c for c, r in book.rules.items() if not r.is_active]
    lines = [
        f"Таблица правил: {book.version}",
        f"  Параметры:   {len(book.params)}",
        f"  Стадии:      {len(book.stages)}  ({', '.join(book.stages)})",
        f"  Переходы:    {len(book.transitions)}",
        f"  Исключения:  {len(book.exceptions)}",
        f"  Правила:     {len(book.rules)}  (активных {len(book.active_rules)}"
        + (f"; выключены: {', '.join(inactive)})" if inactive else ")"),
        f"  Лимиты:      {len(book.limits)}",
        f"  Поля данных: {len(book.field_names)}",
        f"  Примеры:     {len(book.examples)}",
        "",
        "Меры и лимиты:",
    ]
    measures = dict.fromkeys(r.measure for r in book.rules.values())
    for m in measures:
        rules = book.rules_for_measure(m, active_only=False)
        codes = ", ".join(r.code + ("" if r.is_active else " (выкл.)") for r in rules)
        lim = book.limits.get(m)
        lim_text = "нет строки в «Лимитах»" if lim is None else (
            "нет ограничений" if lim.value is None else f"{lim.value} в месяц")
        lines.append(f"  {m:<19} {lim_text:<24} {codes}")
    lines.append("")
    if book.warnings:
        lines.append(f"Предупреждения ({len(book.warnings)}):")
        lines += [f"  - {w}" for w in book.warnings]
    else:
        lines.append("Предупреждений нет.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Проверить и показать таблицу правил воздействия")
    parser.add_argument("path", help="путь к Excel-файлу таблицы правил")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
    try:
        book = load_rulebook(args.path)
    except (RuleBookError, FileNotFoundError) as e:
        print(f"ОШИБКА. {e}", file=sys.stderr)
        return 1
    print(summary(book))
    return 0


if __name__ == "__main__":
    sys.exit(main())
