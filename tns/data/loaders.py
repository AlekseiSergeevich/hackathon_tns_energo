"""Чтение исходных файлов 01–14 в таблицы с типами по docs/interfaces.md.

Все файлы читает одна функция load_all(). Она возвращает словарь таблиц:

    info      — файл 01:    ls, has_phone, has_ebill, has_benefits, not_residing,
                            remote_disconnect, house_type, gasified, address_guid
    osv       — файл 02:    ls, month, saldo_start, accrual, paid
    payments  — файл 03:    ls, payment_date, amount, payment_method
    events    — файлы 04–13: ls, measure, event_date, date_precision
    limits    — файл 14:    measure, limit (<NA> — без ограничений)

Каждый файл проверяется при чтении (tns/data/validator.py): обязательные колонки, типы значений,
пустые и испорченные файлы. Ошибки по всем файлам собираются и выдаются одним ValidationError
с понятным списком, например «В файле «06 СМС ХК.xlsx» нет колонки «Дата».». Предупреждения
(строки, которые не прочитались, повторы) — в table.attrs["problems"] и в лог.

Таблицы 01–13 кэшируются в parquet; кэш сбрасывается, если изменился исходный файл или настройки чтения.

Запуск:
    python -m tns.data.loaders [--data ПАПКА] [--config ФАЙЛ] [--no-cache]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tns.data import validator
from tns.data.validator import ERROR, Problem, ValidationError

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS = REPO_ROOT / "configs" / "settings.yaml"
# Меняйте при любом изменении чтения или типов, чтобы старый кэш не использовался.
LOADER_VERSION = 3

INFO_FILE = "01 Общая информация о ЛС ХК.xlsx"
OSV_FILE = "02 Обортно-сальдовая ведомость ЛС ХК.xlsx"
PAYMENTS_FILE = "03 Оплаты ХК.csv"
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
LIMITS_FILE = "14 Лимиты мер воздействия ХК.xlsx"
# Меры, у которых в данных известен только месяц: дата проставлена 1-м числом.
MONTH_PRECISION_MEASURES = {"autodial", "email", "sms", "claim", "visit"}
DATE_PRECISIONS = ["month", "day"]

OSV_METRICS = {
    "сз на начало": "saldo_start",
    "начислено": "accrual",
    "оплачено": "paid",
    "опалачено": "paid",  # опечатка в шапке файла 02
}
OSV_METRIC_NAMES = {"saldo_start": "СЗ на начало", "accrual": "Начислено", "paid": "Оплачено"}
INFO_FLAGS = {
    "has_phone": "Наличие телефона",
    "has_ebill": "электронная квитанция",
    "has_benefits": "Наличие льгот",
    "not_residing": "не проживает",
    "remote_disconnect": "Возможность дистанционного отключения",
    "gasified": "Газификация дома",
}
HOUSE_TYPES = ["ЧД", "МКД", "Общежитие"]
INFO_GUID_COLUMN = "Адрес (ГУИД)"
NO_LIMIT_TEXT = "нет ограничений"

# Типы колонок по контракту. Список вместо строки — category с этим набором значений.
SCHEMAS: dict[str, dict[str, str | list[str]]] = {
    "info": {
        "ls": "int64",
        **{flag: "boolean" for flag in INFO_FLAGS},
        "house_type": HOUSE_TYPES,
        "address_guid": "string",
    },
    "osv": {"ls": "int64", "month": "datetime64[ns]", "saldo_start": "float64", "accrual": "float64", "paid": "float64"},
    "payments": {"ls": "int64", "payment_date": "datetime64[ns]", "amount": "float64", "payment_method": "Int64"},
    "events": {"ls": "int64", "measure": list(MEASURE_FILES), "event_date": "datetime64[ns]", "date_precision": DATE_PRECISIONS},
    "limits": {"measure": list(MEASURE_FILES), "limit": "Int64"},
    # Выгрузка норматива заказчика (P-07). month = NaT — значение действует для всех месяцев.
    "norm": {"ls": "int64", "month": "datetime64[ns]", "norm": "float64"},
}
TABLE_SOURCES = {
    "info": [INFO_FILE],
    "osv": [OSV_FILE],
    "payments": [PAYMENTS_FILE],
    "events": list(MEASURE_FILES.values()),
}


def load_settings(path: Path | str | None = None) -> dict:
    """Читает configs/settings.yaml."""
    path = Path(path) if path else DEFAULT_SETTINGS
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл настроек: {path}")
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Общие помощники
# ---------------------------------------------------------------------------

def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Не найден входной файл «{path.name}» в папке «{path.parent}».")
    return path


def _read_excel(path: Path, **kwargs) -> pd.DataFrame:
    """pd.read_excel с понятной ошибкой, если файл не открывается."""
    _require_file(path)
    try:
        return pd.read_excel(path, **kwargs)
    except Exception as error:  # noqa: BLE001 — любые сбои разбора файла пересказываем пользователю
        raise ValueError(validator.open_error(path.name, error, "excel")) from error


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    _require_file(path)
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as error:  # noqa: BLE001
        raise ValueError(validator.open_error(path.name, error, "csv")) from error


def _require_columns(columns, required: list[str], file_name: str, hint: str = "") -> None:
    """Ошибка со списком всех отсутствующих колонок и тем, что в файле есть."""
    present = [str(c).strip() for c in columns]
    missing = [c for c in required if c not in present]
    if not missing:
        return
    names = ", ".join(f"«{c}»" for c in missing)
    found = ", ".join(f"«{c}»" for c in present[:10] if c and c.lower() != "nan") or "ничего"
    what = f"нет колонки {names}" if len(missing) == 1 else f"нет колонок {names}"
    raise ValueError(f"В файле «{file_name}» {what}. В шапке найдено: {found}.{hint}")


def _warn(problems: list[Problem], problem: Problem | None) -> None:
    if problem is not None:
        problems.append(problem)
        logger.warning("%s", problem.message)


def _clean_ls(df: pd.DataFrame, column, file_name: str, rows, problems: list[Problem], label: str = "ЛС") -> pd.DataFrame:
    """Приводит номер ЛС к int64. Строки без корректного номера убирает и сообщает об этом."""
    ls = pd.to_numeric(df[column], errors="coerce").astype("float64")
    valid = ls.notna() & (ls % 1 == 0)
    _warn(problems, validator.lost_values(
        file_name, label, ~valid, df[column], rows, what="номер ЛС (целое число)",
    ))
    df = df.loc[valid].copy()
    df["ls"] = ls[valid].astype("int64")
    return df


def _yes_no_to_boolean(values: pd.Series, file_name: str, column: str, rows, problems: list[Problem]) -> pd.Series:
    """«Да» → True, «Нет» → False, пусто → <NA>. Другие значения → <NA> с предупреждением."""
    text = values.astype("string").str.strip().str.lower()
    result = text.map({"да": True, "нет": False}, na_action="ignore").astype("boolean")
    _warn(problems, validator.lost_values(
        file_name, column, result.isna() & text.notna(), values, rows,
        what="«Да» или «Нет»", action="они записаны как неизвестные",
    ))
    return result


def _parse_excel_dates(values: pd.Series) -> pd.Series:
    """Даты из Excel: ячейки с датой как есть, текст — ДД.ММ.ГГГГ или ГГГГ-ММ-ДД. Остальное → NaT."""
    is_date = values.map(lambda v: isinstance(v, datetime)).astype(bool)
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if is_date.any():
        result[is_date] = pd.to_datetime(values[is_date]).astype("datetime64[ns]")
    text = values[~is_date & values.notna()].astype(str).str.strip()
    if len(text):
        parsed = pd.to_datetime(text, format="%d.%m.%Y", errors="coerce")
        parsed = parsed.fillna(pd.to_datetime(text, format="ISO8601", errors="coerce"))
        result[text.index] = parsed.astype("datetime64[ns]")
    return result


# ---------------------------------------------------------------------------
# Файл 01 — анкета ЛС
# ---------------------------------------------------------------------------

def read_info(data_dir: Path) -> pd.DataFrame:
    """Файл 01: флаги «Да/Нет» → boolean, тип дома → category, ГУИД → string."""
    problems: list[Problem] = []
    raw = _read_excel(data_dir / INFO_FILE)
    raw.columns = [str(c).strip() for c in raw.columns]
    columns = [*INFO_FLAGS.values(), *HOUSE_TYPES, INFO_GUID_COLUMN]
    _require_columns(raw.columns, ["ЛС", *columns], INFO_FILE)
    if raw.empty:
        raise ValueError(f"В файле «{INFO_FILE}» нет ни одной строки с данными, только шапка.")

    df = _clean_ls(raw, "ЛС", INFO_FILE, raw.index + 2, problems)[["ls", *columns]]
    duplicated = df.duplicated("ls")
    if duplicated.any():
        conflicts = df.groupby("ls")[columns].nunique(dropna=False).gt(1).any(axis=1).sum()
        differ = (f"у {validator.number(int(conflicts))} ЛС повторы различаются" if conflicts
                  else "содержимое повторов одинаковое")
        _warn(problems, Problem(
            validator.WARNING, INFO_FILE,
            f"В файле «{INFO_FILE}» {validator.repeats(int(duplicated.sum()))} ЛС, оставлена первая строка ({differ}).",
            column="ЛС", count=int(duplicated.sum()), total=len(df), blocking=False,
        ))
        df = df.loc[~duplicated]
    rows = df.index + 2

    result = pd.DataFrame({"ls": df["ls"].astype("int64").to_numpy()})
    for column, source in INFO_FLAGS.items():
        result[column] = _yes_no_to_boolean(df[source], INFO_FILE, source, rows, problems).array

    # Тип дома: первый отмеченный флаг. Если ни один не отмечен — неизвестно.
    marks = pd.concat(
        [_yes_no_to_boolean(df[h], INFO_FILE, h, rows, problems).fillna(False) for h in HOUSE_TYPES], axis=1
    ).to_numpy(bool)
    several = marks.sum(axis=1) > 1
    if several.any():
        _warn(problems, validator.warning(INFO_FILE, (
            f"В файле «{INFO_FILE}» у {validator.number(int(several.sum()))} ЛС отмечено несколько типов дома, "
            "взят первый из «ЧД», «МКД», «Общежитие»."
        )))
    codes = marks.argmax(axis=1)
    codes[~marks.any(axis=1)] = -1
    result["house_type"] = pd.Categorical.from_codes(codes, categories=HOUSE_TYPES)
    result["address_guid"] = df[INFO_GUID_COLUMN].astype("string").str.strip().array
    result.attrs["problems"] = problems
    return result


# ---------------------------------------------------------------------------
# Файл 02 — ОСВ
# ---------------------------------------------------------------------------

def _parse_osv_blocks(header: pd.DataFrame) -> tuple[int, list[tuple[pd.Timestamp, dict]], list[int]]:
    """Разбирает двухуровневую шапку ОСВ.

    Возвращает номер колонки ЛС, блоки (месяц, {показатель: номер колонки}) и индексы блоков,
    которые в шапке повторяют уже встреченный месяц.
    """
    if len(header) < 2:
        raise ValueError(
            f"В файле «{OSV_FILE}» нет двухстрочной шапки: в первой строке «ЛС» и месяцы, "
            "во второй — «СЗ на начало», «Начислено», «Оплачено»."
        )
    top, sub = header.iloc[0].tolist(), header.iloc[1].tolist()
    ls_columns = [i for i, value in enumerate(top) if str(value).strip() == "ЛС"]
    if not ls_columns:
        raise ValueError(f"В файле «{OSV_FILE}» нет колонки «ЛС» в первой строке шапки.")

    blocks: list[tuple[pd.Timestamp, dict]] = []
    for i, value in enumerate(top):
        if i == ls_columns[0]:
            continue
        if pd.notna(value):
            month = pd.to_datetime(value, errors="coerce")
            if pd.isna(month):
                raise ValueError(
                    f"В файле «{OSV_FILE}», колонка {i + 1}: «{value}» в первой строке шапки не похоже на месяц."
                )
            blocks.append((month.to_period("M").to_timestamp(), {}))
        metric = OSV_METRICS.get(str(sub[i]).strip().lower())
        if metric and blocks:
            blocks[-1][1].setdefault(metric, i)

    if not blocks:
        raise ValueError(f"В файле «{OSV_FILE}» в шапке не найдено ни одного месяца.")
    for month, metrics in blocks:
        missing = [OSV_METRIC_NAMES[m] for m in OSV_METRIC_NAMES if m not in metrics]
        if missing:
            raise ValueError(
                f"В файле «{OSV_FILE}» для месяца {month:%Y-%m} нет "
                f"{'колонки' if len(missing) == 1 else 'колонок'} {', '.join(f'«{m}»' for m in missing)}."
            )

    seen: set[pd.Timestamp] = set()
    repeated = []
    for index, (month, _) in enumerate(blocks):
        if month in seen:
            repeated.append(index)
        seen.add(month)
    return ls_columns[0], blocks, repeated


def read_osv(data_dir: Path, settings: dict) -> pd.DataFrame:
    """Файл 02 в длинном формате: ls, month, saldo_start, accrual, paid."""
    problems: list[Problem] = []
    path = data_dir / OSV_FILE
    policy = settings["osv"]["duplicate_month"]
    if policy not in {"shift_next", "drop", "error"}:
        raise ValueError(f"Неизвестное значение osv.duplicate_month: «{policy}». Допустимо: shift_next, drop, error.")
    header = _read_excel(path, header=None, nrows=2)
    ls_column, blocks, repeated = _parse_osv_blocks(header)

    body = _read_excel(path, header=None, skiprows=2)
    if body.empty:
        raise ValueError(f"В файле «{OSV_FILE}» нет ни одной строки с данными, только шапка.")
    body = _clean_ls(body, ls_column, OSV_FILE, body.index + 3, problems)
    duplicated = body.duplicated("ls")
    if duplicated.any():
        _warn(problems, Problem(
            validator.WARNING, OSV_FILE,
            f"В файле «{OSV_FILE}» {validator.repeats(int(duplicated.sum()))} ЛС, оставлена первая строка.",
            column="ЛС", count=int(duplicated.sum()), total=len(body), blocking=False,
        ))
        body = body.loc[~duplicated]

    known_months = {month for month, _ in blocks}
    drop_blocks = set()
    for index in repeated:
        month, metrics = blocks[index]
        previous = next(m for mo, m in blocks[:index] if mo == month)
        before = {k: pd.to_numeric(body[c], errors="coerce") for k, c in previous.items()}
        expected = before["saldo_start"] + before["accrual"] - before["paid"]
        actual = pd.to_numeric(body[metrics["saldo_start"]], errors="coerce")
        nonzero = (actual != 0) | (expected != 0)
        share = (expected - actual).abs().lt(0.01)[nonzero].mean() if nonzero.any() else float("nan")
        if policy == "error":
            raise ValueError(
                f"В файле «{OSV_FILE}» месяц {month:%Y-%m} повторяется в шапке (колонки с {min(metrics.values()) + 1}). "
                "Исправьте шапку или задайте osv.duplicate_month: shift_next / drop."
            )
        action = "блок пропущен" if policy == "drop" else f"блок считается месяцем {month + pd.DateOffset(months=1):%Y-%m}"
        _warn(problems, validator.warning(OSV_FILE, (
            f"В файле «{OSV_FILE}» месяц {month:%Y-%m} повторяется в шапке (колонки с {min(metrics.values()) + 1}); "
            f"{action} (настройка osv.duplicate_month: {policy}). У {validator.percent(share)} ненулевых строк "
            "сальдо этого блока продолжает предыдущий блок."
        )))
        if policy == "drop":
            drop_blocks.add(index)
            continue
        next_month = month + pd.DateOffset(months=1)
        if next_month in known_months:
            raise ValueError(
                f"В файле «{OSV_FILE}» месяц {month:%Y-%m} повторяется, а следующий месяц "
                f"{next_month:%Y-%m} уже есть в шапке: сдвинуть блок нельзя. Исправьте шапку файла."
            )
        known_months.add(next_month)
        blocks[index] = (next_month, metrics)

    kept = [(month, metrics) for index, (month, metrics) in enumerate(blocks) if index not in drop_blocks]
    frames = []
    for month, metrics in kept:
        frame = pd.DataFrame({"ls": body["ls"].to_numpy(), "month": month})
        for metric, column in metrics.items():
            frame[metric] = pd.to_numeric(body[column], errors="coerce").astype("float64").to_numpy()
        frames.append(frame)
    osv = pd.concat(frames, ignore_index=True)
    osv["month"] = osv["month"].astype("datetime64[ns]")

    # Пустые и нечисловые суммы: по одной записи на показатель за все месяцы.
    rows = body.index + 3
    for metric, name in OSV_METRIC_NAMES.items():
        values = pd.concat([body[metrics[metric]] for _, metrics in kept], ignore_index=True)
        _warn(problems, validator.lost_values(
            OSV_FILE, name, osv[metric].isna(), values, np.tile(rows, len(kept)),
            what="сумма в рублях", action="значения оставлены пустыми",
        ))

    months = pd.date_range(osv["month"].min(), osv["month"].max(), freq="MS")
    if len(months) != osv["month"].nunique():
        absent = [f"{m:%Y-%m}" for m in months if m not in set(osv["month"])]
        raise ValueError(f"В файле «{OSV_FILE}» в шапке пропущены месяцы: {', '.join(absent)}.")
    osv = osv.sort_values(["ls", "month"], ignore_index=True)
    osv.attrs["problems"] = problems
    return osv


# ---------------------------------------------------------------------------
# Файл 03 — оплаты
# ---------------------------------------------------------------------------

def read_payments(data_dir: Path) -> pd.DataFrame:
    """Файл 03: ls, payment_date, amount, payment_method (код способа оплаты, «NULL» → <NA>).

    Одинаковые строки не удаляем: ОСВ сходится с суммой вместе с ними, то есть это
    отдельные платежи. Отрицательные суммы (сторно) тоже остаются.
    """
    problems: list[Problem] = []
    raw = _read_csv(data_dir / PAYMENTS_FILE, sep=";", encoding="utf-8-sig", dtype=str)
    raw.columns = [str(c).strip() for c in raw.columns]
    hint = ""
    if len(raw.columns) == 1 and any(sep in raw.columns[0] for sep in ",\t"):
        hint = " Похоже, колонки разделены не точкой с запятой: в файле должен быть разделитель «;»."
    _require_columns(raw.columns, ["Номер", "Дата оплаты", "Сумма", "Способ оплаты"], PAYMENTS_FILE, hint)
    if raw.empty:
        raise ValueError(f"В файле «{PAYMENTS_FILE}» нет ни одной строки с оплатами, только шапка.")

    df = _clean_ls(raw, "Номер", PAYMENTS_FILE, raw.index + 2, problems, label="Номер")
    rows = df.index + 2
    df["payment_date"] = pd.to_datetime(df["Дата оплаты"].str.strip(), format="%d.%m.%Y", errors="coerce")
    amount = df["Сумма"].str.replace(r"[\s ]", "", regex=True).str.replace(",", ".", regex=False)
    df["amount"] = pd.to_numeric(amount, errors="coerce").astype("float64")
    _warn(problems, validator.lost_values(
        PAYMENTS_FILE, "Дата оплаты", df["payment_date"].isna(), df["Дата оплаты"], rows, what="дата ДД.ММ.ГГГГ",
    ))
    _warn(problems, validator.lost_values(
        PAYMENTS_FILE, "Сумма", df["amount"].isna(), df["Сумма"], rows, what="число, например 1240,50",
    ))
    valid = df["payment_date"].notna() & df["amount"].notna()
    df = df.loc[valid]

    method_text = df["Способ оплаты"].str.strip().replace({"NULL": None, "": None})
    method = pd.to_numeric(method_text, errors="coerce")
    _warn(problems, validator.lost_values(
        PAYMENTS_FILE, "Способ оплаты", method.isna() & method_text.notna(), df["Способ оплаты"], df.index + 2,
        what="числовой код или NULL", blocking=False, action="записаны как неизвестные",
    ))

    payments = pd.DataFrame({
        "ls": df["ls"].to_numpy(),
        "payment_date": df["payment_date"].astype("datetime64[ns]").to_numpy(),
        "amount": df["amount"].to_numpy(),
        "payment_method": method.astype("Int64").array,
    })
    payments.attrs["problems"] = problems
    return payments


# ---------------------------------------------------------------------------
# Выгрузка норматива заказчика (P-07)
# ---------------------------------------------------------------------------

NORM_COLUMNS_DEFAULT = {"ls": "ЛС", "norm": "Норматив", "month": "Месяц"}


def _empty_norm() -> pd.DataFrame:
    return pd.DataFrame({
        "ls": pd.Series(dtype="int64"), "month": pd.Series(dtype="datetime64[ns]"), "norm": pd.Series(dtype="float64"),
    })


def customer_norm_path(data_dir: Path, settings: dict) -> Path | None:
    """Путь к выгрузке норматива из norm.customer_file. Относительный путь — от папки с данными."""
    name = settings.get("norm", {}).get("customer_file")
    if not name:
        return None
    path = Path(name)
    return path if path.is_absolute() else Path(data_dir) / path


def read_customer_norm(data_dir: Path, settings: dict) -> pd.DataFrame:
    """Выгрузка норматива заказчика: ls, month, norm (одинарный, руб.).

    Файл .xlsx или .csv (разделитель «;» или «,», десятичная запятая допускается). Названия колонок —
    norm.customer_columns. Колонка месяца необязательна: без неё значение действует для всех месяцев.
    Значение делится на norm.customer_multiplier (2 — если в выгрузке сразу двукратный норматив).
    Если путь не задан (norm.customer_file: null), возвращает пустую таблицу: работает прокси.
    """
    path = customer_norm_path(data_dir, settings)
    if path is None:
        return _empty_norm()
    options = settings["norm"]
    columns = {**NORM_COLUMNS_DEFAULT, **(options.get("customer_columns") or {})}
    multiplier = options.get("customer_multiplier", 1)
    if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)) or multiplier <= 0:
        raise ValueError(f"norm.customer_multiplier — положительное число (1 или 2), а указано «{multiplier}».")

    file_name, problems = path.name, []
    if path.suffix.lower() == ".csv":
        raw = _read_csv(path, sep=None, engine="python", encoding="utf-8-sig", dtype=str)
    else:
        raw = _read_excel(path, dtype=object)
    raw.columns = [str(c).strip() for c in raw.columns]
    has_month = columns["month"] in raw.columns
    _require_columns(raw.columns, [columns["ls"], columns["norm"]], file_name,
                     " Названия колонок задаются в configs/settings.yaml, раздел norm.customer_columns.")
    if raw.empty:
        raise ValueError(f"В файле «{file_name}» нет ни одной строки с нормативом, только шапка.")

    df = _clean_ls(raw, columns["ls"], file_name, raw.index + 2, problems, label=columns["ls"])
    rows = df.index + 2
    text = df[columns["norm"]].astype("string").str.replace(r"[\s\u00a0]", "", regex=True).str.replace(",", ".", regex=False)
    value = pd.to_numeric(text, errors="coerce").astype("float64")
    bad = value.isna() | (value <= 0)
    _warn(problems, validator.lost_values(
        file_name, columns["norm"], bad, df[columns["norm"]], rows, what="положительное число, руб.",
    ))
    if has_month:
        month = _parse_excel_dates(df[columns["month"]]).dt.to_period("M").dt.to_timestamp()
        no_month = month.isna() & df[columns["month"]].notna()
        _warn(problems, validator.lost_values(
            file_name, columns["month"], no_month, df[columns["month"]], rows, what="дата или месяц, например 01.03.2026",
        ))
        bad |= no_month
    else:
        month = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    norm = pd.DataFrame({
        "ls": df["ls"].to_numpy(),
        "month": month.astype("datetime64[ns]").to_numpy(),
        "norm": (value / multiplier).to_numpy(),
    })[~bad.to_numpy()].reset_index(drop=True)
    repeated = norm.duplicated(["ls", "month"], keep=False)
    if repeated.any():
        example = norm.loc[repeated].iloc[0]
        when = "" if pd.isna(example["month"]) else f" за {example['month']:%Y-%m}"
        raise ValueError(
            f"В файле «{file_name}» у {norm.loc[repeated, 'ls'].nunique()} ЛС норматив указан несколько раз "
            f"на один месяц, например ЛС {example['ls']}{when}. Оставьте одно значение."
        )
    norm.attrs["problems"] = problems
    logger.info("Норматив заказчика: %d строк, %d ЛС из «%s».", len(norm), norm["ls"].nunique(), file_name)
    return norm


# ---------------------------------------------------------------------------
# Файлы 04–13 — меры
# ---------------------------------------------------------------------------

EVENT_COLUMNS = ["ls", "event_date"]


def read_measure_file(data_dir: Path, measure: str) -> tuple[pd.DataFrame, list[Problem]]:
    """Один файл мер: ls, event_date. Над шапкой «ЛС | Дата» может быть строка с названием меры."""
    file_name = MEASURE_FILES[measure]
    problems: list[Problem] = []
    raw = _read_excel(data_dir / file_name, header=None, dtype=object)
    if raw.empty:
        raise ValueError(f"Файл «{file_name}» пуст: нужна шапка «ЛС | Дата» и строки с мерами.")
    header_row = next(
        (i for i in range(min(5, len(raw))) if "ЛС" in {str(v).strip() for v in raw.iloc[i] if pd.notna(v)}),
        1 if len(raw) > 1 else 0,
    )
    columns = [str(v).strip() for v in raw.iloc[header_row]]
    _require_columns(columns, ["ЛС", "Дата"], file_name)
    data = raw.iloc[header_row + 1:].copy()
    data.columns = columns
    data = data.loc[:, ~pd.Index(columns).duplicated()][["ЛС", "Дата"]]
    if data.empty:
        _warn(problems, validator.warning(file_name, f"В файле «{file_name}» нет ни одной строки с мерами."))
        return pd.DataFrame({"ls": pd.Series(dtype="int64"), "event_date": pd.Series(dtype="datetime64[ns]")}), problems

    rows = data.index + 1  # header=None: индекс строки + 1 = номер строки в Excel
    df = _clean_ls(data, "ЛС", file_name, rows, problems)
    df["event_date"] = _parse_excel_dates(df["Дата"])
    _warn(problems, validator.lost_values(
        file_name, "Дата", df["event_date"].isna(), df["Дата"], df.index + 1, what="дата",
    ))
    df = df.loc[df["event_date"].notna(), EVENT_COLUMNS]

    if measure in MONTH_PRECISION_MEASURES:
        not_first = int((df["event_date"].dt.day != 1).sum())
        if not_first:
            _warn(problems, validator.warning(file_name, (
                f"В файле «{file_name}» {validator.number(not_first)} дат не на 1-е число, "
                "хотя для этой меры обычно известен только месяц."
            ), "Дата"))
    repeats = int(df.duplicated().sum())
    if repeats:
        logger.info("Файл «%s»: %d повторов (ЛС, дата), оставлены для подсчёта n_raw.", file_name, repeats)
    return df.astype({"ls": "int64", "event_date": "datetime64[ns]"}), problems


def read_measure_events(data_dir: Path) -> pd.DataFrame:
    """Файлы 04–13: ls, measure, event_date, date_precision.

    Ошибки по всем десяти файлам собираются в один ValidationError. Повторы строк не удаляем:
    их число нужно для n_raw в журнале мер (docs/interfaces.md, раздел 3).
    """
    problems: list[Problem] = []
    frames = []
    for measure, file_name in MEASURE_FILES.items():
        try:
            df, file_problems = read_measure_file(data_dir, measure)
        except (OSError, ValueError) as error:
            problems.append(validator.error(file_name, str(error)))
            continue
        problems += file_problems
        frames.append(df.assign(measure=measure))
    if any(p.level == ERROR for p in problems):
        raise ValidationError(problems)

    events = pd.concat(frames, ignore_index=True)
    events["measure"] = pd.Categorical(events["measure"], categories=list(MEASURE_FILES))
    events["event_date"] = events["event_date"].astype("datetime64[ns]")
    events["date_precision"] = pd.Categorical(
        events["measure"].astype(str).map(lambda m: "month" if m in MONTH_PRECISION_MEASURES else "day"),
        categories=DATE_PRECISIONS,
    )
    events = events[["ls", "measure", "event_date", "date_precision"]].sort_values(
        ["ls", "measure", "event_date"], ignore_index=True
    )
    events.attrs["problems"] = problems
    return events


# ---------------------------------------------------------------------------
# Файл 14 — лимиты
# ---------------------------------------------------------------------------

def _limits_frame(limits: dict[str, int | None], problems: list[Problem]) -> pd.DataFrame:
    frame = pd.DataFrame({
        "measure": pd.Categorical(list(MEASURE_FILES), categories=list(MEASURE_FILES)),
        "limit": pd.array([limits[m] for m in MEASURE_FILES], dtype="Int64"),
    })
    frame.attrs["problems"] = problems
    return frame


def _read_limits_yaml(path: Path) -> dict[str, int | None]:
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл запасных лимитов «{path}».")
    with open(path, encoding="utf-8") as fh:
        limits = (yaml.safe_load(fh) or {}).get("limits") or {}
    missing = [m for m in MEASURE_FILES if m not in limits]
    if missing:
        raise ValueError(f"В файле «{path.name}» нет лимитов для мер: {', '.join(missing)}.")
    return {m: None if limits[m] is None else int(limits[m]) for m in MEASURE_FILES}


def read_limits(data_dir: Path, settings: dict) -> pd.DataFrame:
    """Файл 14: measure, limit. «нет ограничений» → <NA>.

    Меры сопоставляются по номеру файла в начале названия («04 Автодозвон ХК» → autodial).
    Если файла 14 нет, лимиты берутся из запасного limits.yaml. Расхождения с ним — предупреждение.
    """
    problems: list[Problem] = []
    fallback = REPO_ROOT / settings["paths"]["limits_fallback"]
    path = data_dir / LIMITS_FILE
    if not path.exists():
        _warn(problems, validator.warning(LIMITS_FILE, (
            f"Не найден файл «{LIMITS_FILE}», лимиты взяты из «{fallback.name}»."
        )))
        return _limits_frame(_read_limits_yaml(fallback), problems)

    raw = _read_excel(path, header=None, dtype=str)
    by_number = {name[:2]: measure for measure, name in MEASURE_FILES.items()}
    limits: dict[str, int | None] = {}
    errors = []
    for row, (name, value) in enumerate(raw.iloc[1:, :2].itertuples(index=False), start=2):
        if pd.isna(name) or not str(name).strip():
            continue
        found = re.match(r"\s*(\d{2})\b", str(name))
        measure = by_number.get(found.group(1)) if found else None
        if measure is None:
            errors.append(f"В файле «{LIMITS_FILE}», строка {row}: неизвестная мера «{name}». "
                          "Название должно начинаться с номера файла меры, например «06 СМС ХК».")
            continue
        text = "" if pd.isna(value) else str(value).strip()
        if text.lower() == NO_LIMIT_TEXT:
            limits[measure] = None
        elif re.fullmatch(r"\d+(\.0+)?", text):
            limits[measure] = int(float(text))
        else:
            errors.append(f"В файле «{LIMITS_FILE}», строка {row}: для меры «{name}» лимит "
                          f"«{text or 'пусто'}», а нужно целое число или «{NO_LIMIT_TEXT}».")
    missing = [MEASURE_FILES[m].removesuffix(".xlsx") for m in MEASURE_FILES if m not in limits]
    if missing:
        errors.append(f"В файле «{LIMITS_FILE}» нет лимитов для мер: {', '.join(missing)}.")
    if errors:
        raise ValidationError([validator.error(LIMITS_FILE, e) for e in errors])

    if fallback.exists():
        reserve = _read_limits_yaml(fallback)
        differ = [f"{m}: {limits[m]} против {reserve[m]}" for m in MEASURE_FILES if limits[m] != reserve[m]]
        if differ:
            _warn(problems, validator.warning(LIMITS_FILE, (
                f"Лимиты в файле «{LIMITS_FILE}» отличаются от «{fallback.name}» (используется файл 14): "
                f"{'; '.join(differ)}."
            )))
    return _limits_frame(limits, problems)


# ---------------------------------------------------------------------------
# Проверка типов и кэш
# ---------------------------------------------------------------------------

def check_schema(name: str, df: pd.DataFrame, schemas: dict | None = None, keys: list[str] | None = None) -> None:
    """Проверяет колонки, их порядок и типы таблицы по схеме (по умолчанию SCHEMAS) и уникальность keys."""
    schema = (SCHEMAS if schemas is None else schemas)[name]
    problems = []
    for column, expected in schema.items():
        if column not in df.columns:
            problems.append(f"нет колонки «{column}»")
            continue
        dtype = df[column].dtype
        if isinstance(expected, list):
            if not isinstance(dtype, pd.CategoricalDtype) or list(dtype.categories) != expected:
                problems.append(f"«{column}»: ожидалась category {expected}, получено {dtype}")
        elif str(dtype) != expected:
            problems.append(f"«{column}»: ожидался тип {expected}, получено {dtype}")
    extra = [c for c in df.columns if c not in schema]
    if extra:
        problems.append(f"лишние колонки: {', '.join(extra)}")
    elif list(df.columns) != [c for c in schema if c in df.columns]:
        problems.append(f"порядок колонок не как в контракте: {', '.join(schema)}")
    if "ls" in df.columns and df["ls"].isna().any():
        problems.append("есть пустые ЛС")
    if keys and not problems and df.duplicated(keys).any():
        problems.append(f"повторяются ключи ({', '.join(keys)}): {int(df.duplicated(keys).sum())} строк")
    if problems:
        raise ValueError(f"Таблица «{name}» не соответствует docs/interfaces.md: {'; '.join(problems)}.")


def _cache_key(name: str, data_dir: Path, settings: dict) -> dict:
    files = {}
    for file_name in TABLE_SOURCES[name]:
        stat = _require_file(data_dir / file_name).stat()
        files[file_name] = [stat.st_mtime_ns, stat.st_size]
    options = {"duplicate_month": settings["osv"]["duplicate_month"]} if name == "osv" else {}
    return {"version": LOADER_VERSION, "data_dir": str(data_dir.resolve()), "files": files, "options": options}


def _read_table(name: str, data_dir: Path, settings: dict) -> pd.DataFrame:
    if name == "info":
        return read_info(data_dir)
    if name == "osv":
        return read_osv(data_dir, settings)
    if name == "payments":
        return read_payments(data_dir)
    return read_measure_events(data_dir)


def _load_cached(name: str, data_dir: Path, settings: dict, cache_dir: Path | None) -> pd.DataFrame:
    if cache_dir is None:
        table = _read_table(name, data_dir, settings)
        table.attrs["cached"] = False
        return table

    key = _cache_key(name, data_dir, settings)
    table_path, meta_path = cache_dir / f"{name}.parquet", cache_dir / f"{name}.json"
    if table_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("key") == key:
                table = pd.read_parquet(table_path)
                table.attrs = {"problems": [Problem.from_dict(p) for p in meta.get("problems", [])], "cached": True}
                logger.info("Таблица «%s» взята из кэша.", name)
                return table
        except (OSError, ValueError, TypeError) as error:
            logger.warning("Кэш таблицы «%s» не прочитан (%s), читаю исходные файлы.", name, error)

    table = _read_table(name, data_dir, settings)
    cache_dir.mkdir(parents=True, exist_ok=True)
    problems = table.attrs.get("problems", [])
    table.attrs = {}
    table.to_parquet(table_path, index=False)
    meta = {"key": key, "problems": [p.to_dict() for p in problems]}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    table.attrs = {"problems": problems, "cached": False}
    return table


def load_all(data_dir: Path | str, settings: dict, cache_dir: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Читает и проверяет файлы 01–14, возвращает таблицы info, osv, payments, events, limits, norm.

    norm — выгрузка норматива заказчика (norm.customer_file), пустая, если её нет.

    Если хоть в одном файле ошибка, бросает ValidationError со списком ошибок по всем файлам.
    Предупреждения — в table.attrs["problems"]. cache_dir — папка кэша для 01–13, None — без кэша.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Не найдена папка с данными: «{data_dir}».")
    cache_dir = Path(cache_dir) if cache_dir is not None else None

    tables: dict[str, pd.DataFrame] = {}
    problems: list[Problem] = []
    readers = {name: (lambda n=name: _load_cached(n, data_dir, settings, cache_dir)) for name in TABLE_SOURCES}
    readers["limits"] = lambda: read_limits(data_dir, settings)
    readers["norm"] = lambda: read_customer_norm(data_dir, settings)
    for name, read in readers.items():
        logger.info("Читаю «%s».", name)
        try:
            tables[name] = read()
        except ValidationError as error:
            problems += error.problems
            continue
        except (OSError, ValueError) as error:
            norm_path = customer_norm_path(data_dir, settings)
            default = [norm_path.name if norm_path else ""] if name == "norm" else [LIMITS_FILE]
            sources = TABLE_SOURCES.get(name, default)
            problems.append(validator.error(sources[0] if len(sources) == 1 else "", str(error)))
            continue
        problems += tables[name].attrs.get("problems", [])
        try:
            check_schema(name, tables[name])
        except ValueError as error:
            problems.append(validator.error("", str(error)))

    problems = validator.escalate(problems, settings["validation"]["max_lost_share"])
    if any(p.level == ERROR for p in problems):
        raise ValidationError(problems, data_dir)
    return tables


def default_cache_dir(settings: dict) -> Path:
    return REPO_ROOT / settings["paths"]["cache_dir"]


# ---------------------------------------------------------------------------
# Командная строка: проверка и сводка по таблицам
# ---------------------------------------------------------------------------

def summarize(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """По каждой таблице: строк, ЛС, мин./макс. дата, число предупреждений."""
    rows = []
    for name, table in tables.items():
        dates = [c for c in table.columns if pd.api.types.is_datetime64_any_dtype(table[c])]
        known = table[dates[0]].dropna() if dates else pd.Series(dtype="datetime64[ns]")
        rows.append({
            "таблица": name,
            "строк": len(table),
            "ЛС": table["ls"].nunique() if "ls" in table.columns else None,
            "с": f"{known.min():%Y-%m-%d}" if len(known) else "",
            "по": f"{known.max():%Y-%m-%d}" if len(known) else "",
            "предупреждений": len(table.attrs.get("problems", [])),
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Проверяет и читает файлы 01–14, печатает сводку по таблицам.")
    parser.add_argument("--config", help="файл настроек (по умолчанию configs/settings.yaml)")
    parser.add_argument("--data", help="папка с файлами 01–14 (по умолчанию из настроек)")
    parser.add_argument("--no-cache", action="store_true", help="читать исходные файлы, не используя кэш")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        settings = load_settings(args.config)
        cache_dir = None if args.no_cache else default_cache_dir(settings)
        tables = load_all(args.data or REPO_ROOT / settings["paths"]["data_dir"], settings, cache_dir)
    except (OSError, ValueError, KeyError) as error:
        logger.error("Файлы не прочитаны. %s", error)
        return 1

    with pd.option_context("display.max_colwidth", None, "display.width", 200):
        print(summarize(tables).to_string(index=False))
    print()
    print(tables["limits"].to_string(index=False))
    warnings = [p for table in tables.values() for p in table.attrs.get("problems", [])]
    if warnings:
        print(f"\nПредупреждения ({len(warnings)}):")
        for problem in warnings:
            print(f"  - {problem.message}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
