"""Чтение исходных файлов 01–14 в таблицы с типами по docs/interfaces.md.

Все файлы читает одна функция load_all(). Она возвращает словарь таблиц:

    info      — файл 01:    ls, has_phone, has_ebill, has_benefits, not_residing,
                            remote_disconnect, house_type, gasified, address_guid
    osv       — файл 02:    ls, month, saldo_start, accrual, paid
    payments  — файл 03:    ls, payment_date, amount, payment_method
    events    — файлы 04–13: ls, measure, event_date, date_precision
    limits    — файл 14:    measure, limit (<NA> — без ограничений)

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
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS = REPO_ROOT / "configs" / "settings.yaml"
# Меняйте при любом изменении чтения или типов, чтобы старый кэш не использовался.
LOADER_VERSION = 1

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
        raise FileNotFoundError(f"Не найден входной файл: {path}")
    return path


def _require_columns(df: pd.DataFrame, columns: list[str], file_name: str) -> None:
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"В файле «{file_name}» нет колонки «{column}».")


def _clean_ls(df: pd.DataFrame, column, file_name: str, issues: dict) -> pd.DataFrame:
    """Приводит номер ЛС к int64. Строки без корректного номера убирает и пишет об этом в лог."""
    ls = pd.to_numeric(df[column], errors="coerce").astype("float64")
    valid = ls.notna() & (ls % 1 == 0)
    if not valid.all():
        issues["строк без номера ЛС"] = int((~valid).sum())
        logger.warning("Файл «%s»: убрано %d строк без номера ЛС.", file_name, int((~valid).sum()))
    df = df.loc[valid].copy()
    df["ls"] = ls[valid].astype("int64")
    return df


def _yes_no_to_boolean(values: pd.Series, column: str, issues: dict) -> pd.Series:
    """«Да» → True, «Нет» → False, остальное → <NA> (с предупреждением, если значение непустое)."""
    text = values.astype("string").str.strip().str.lower()
    result = text.map({"да": True, "нет": False}, na_action="ignore").astype("boolean")
    unknown = result.isna() & text.notna()
    if unknown.any():
        issues[f"нераспознанных значений «{column}»"] = int(unknown.sum())
        logger.warning(
            "Файл «%s», колонка «%s»: %d значений не «Да»/«Нет», записаны как пустые (например «%s»).",
            INFO_FILE, column, int(unknown.sum()), text[unknown].iloc[0],
        )
    return result


# ---------------------------------------------------------------------------
# Файл 01 — анкета ЛС
# ---------------------------------------------------------------------------

def read_info(data_dir: Path) -> pd.DataFrame:
    """Файл 01: флаги «Да/Нет» → boolean, тип дома → category, ГУИД → string."""
    issues: dict = {}
    raw = pd.read_excel(_require_file(data_dir / INFO_FILE))
    raw.columns = [str(c).strip() for c in raw.columns]
    columns = [*INFO_FLAGS.values(), *HOUSE_TYPES, INFO_GUID_COLUMN]
    _require_columns(raw, ["ЛС", *columns], INFO_FILE)

    df = _clean_ls(raw, "ЛС", INFO_FILE, issues)[["ls", *columns]]
    duplicated = df.duplicated("ls")
    if duplicated.any():
        conflicts = df.groupby("ls")[columns].nunique(dropna=False).gt(1).any(axis=1).sum()
        issues["повторов ЛС"] = int(duplicated.sum())
        logger.warning(
            "Файл «%s»: %d повторов ЛС, оставлена первая строка (нужные поля различаются у %d ЛС).",
            INFO_FILE, int(duplicated.sum()), int(conflicts),
        )
        df = df.loc[~duplicated]
    df = df.reset_index(drop=True)

    result = pd.DataFrame({"ls": df["ls"].astype("int64")})
    for column, source in INFO_FLAGS.items():
        result[column] = _yes_no_to_boolean(df[source], source, issues)

    # Тип дома: первый отмеченный флаг. Если ни один не отмечен — неизвестно.
    marks = pd.concat([_yes_no_to_boolean(df[h], h, issues).fillna(False) for h in HOUSE_TYPES], axis=1).to_numpy(bool)
    several = marks.sum(axis=1) > 1
    if several.any():
        logger.warning("Файл «%s»: у %d ЛС отмечено несколько типов дома, взят первый.", INFO_FILE, int(several.sum()))
    codes = marks.argmax(axis=1)
    codes[~marks.any(axis=1)] = -1
    result["house_type"] = pd.Categorical.from_codes(codes, categories=HOUSE_TYPES)
    result["address_guid"] = df[INFO_GUID_COLUMN].astype("string").str.strip()
    result.attrs["issues"] = issues
    return result


# ---------------------------------------------------------------------------
# Файл 02 — ОСВ
# ---------------------------------------------------------------------------

def _parse_osv_blocks(header: pd.DataFrame) -> tuple[int, list[tuple[pd.Timestamp, dict]], list[int]]:
    """Разбирает двухуровневую шапку ОСВ.

    Возвращает номер колонки ЛС, блоки (месяц, {показатель: номер колонки}) и индексы блоков,
    которые в шапке повторяют уже встреченный месяц.
    """
    top, sub = header.iloc[0].tolist(), header.iloc[1].tolist()
    ls_columns = [i for i, value in enumerate(top) if str(value).strip() == "ЛС"]
    if not ls_columns:
        raise ValueError(f"В файле «{OSV_FILE}» нет колонки «ЛС».")

    blocks: list[tuple[pd.Timestamp, dict]] = []
    for i, value in enumerate(top):
        if i == ls_columns[0]:
            continue
        if pd.notna(value):
            month = pd.to_datetime(value, errors="coerce")
            if pd.isna(month):
                raise ValueError(f"В файле «{OSV_FILE}», колонка {i + 1}: «{value}» не похоже на месяц.")
            blocks.append((month.to_period("M").to_timestamp(), {}))
        metric = OSV_METRICS.get(str(sub[i]).strip().lower())
        if metric and blocks:
            blocks[-1][1].setdefault(metric, i)

    if not blocks:
        raise ValueError(f"В файле «{OSV_FILE}» в шапке не найдено ни одного месяца.")
    for month, metrics in blocks:
        missing = {"saldo_start", "accrual", "paid"} - set(metrics)
        if missing:
            raise ValueError(
                f"В файле «{OSV_FILE}» для месяца {month:%Y-%m} нет показателей: {', '.join(sorted(missing))}."
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
    issues: dict = {}
    path = _require_file(data_dir / OSV_FILE)
    policy = settings["osv"]["duplicate_month"]
    if policy not in {"shift_next", "drop", "error"}:
        raise ValueError(f"Неизвестное значение osv.duplicate_month: «{policy}».")
    header = pd.read_excel(path, header=None, nrows=2)
    ls_column, blocks, repeated = _parse_osv_blocks(header)

    body = pd.read_excel(path, header=None, skiprows=2)
    body = _clean_ls(body, ls_column, OSV_FILE, issues)
    duplicated = body.duplicated("ls")
    if duplicated.any():
        issues["повторов ЛС"] = int(duplicated.sum())
        logger.warning("Файл «%s»: %d повторов ЛС, оставлена первая строка.", OSV_FILE, int(duplicated.sum()))
        body = body.loc[~duplicated]

    known_months = {month for month, _ in blocks}
    drop_blocks = set()
    for index in repeated:
        month, metrics = blocks[index]
        previous = next(m for mo, m in blocks[:index] if mo == month)
        expected = body[previous["saldo_start"]] + body[previous["accrual"]] - body[previous["paid"]]
        actual = body[metrics["saldo_start"]]
        nonzero = (actual != 0) | (expected != 0)
        share = (expected - actual).abs().lt(0.01)[nonzero].mean() if nonzero.any() else float("nan")
        logger.warning(
            "Файл «%s»: месяц %s повторяется в шапке (колонки с %d). У %.1f %% ненулевых строк "
            "сальдо блока продолжает предыдущий блок. Настройка osv.duplicate_month: %s.",
            OSV_FILE, f"{month:%Y-%m}", min(metrics.values()) + 1, share * 100, policy,
        )
        if policy == "error":
            raise ValueError(f"В файле «{OSV_FILE}» месяц {month:%Y-%m} повторяется в шапке.")
        if policy == "drop":
            drop_blocks.add(index)
            continue
        next_month = month + pd.DateOffset(months=1)
        if next_month in known_months:
            raise ValueError(
                f"В файле «{OSV_FILE}» месяц {month:%Y-%m} повторяется, а следующий месяц "
                f"{next_month:%Y-%m} уже есть в шапке: сдвинуть блок нельзя."
            )
        known_months.add(next_month)
        blocks[index] = (next_month, metrics)

    frames = []
    for index, (month, metrics) in enumerate(blocks):
        if index in drop_blocks:
            continue
        frame = pd.DataFrame({"ls": body["ls"], "month": month})
        for metric, column in metrics.items():
            frame[metric] = pd.to_numeric(body[column], errors="coerce").astype("float64")
        frames.append(frame)
    osv = pd.concat(frames, ignore_index=True)
    osv["month"] = osv["month"].astype("datetime64[ns]")

    lost = int(osv[["saldo_start", "accrual", "paid"]].isna().sum().sum())
    if lost:
        issues["значений не прочитано как число"] = lost
        logger.warning("Файл «%s»: %d значений не удалось прочитать как число.", OSV_FILE, lost)
    months = pd.date_range(osv["month"].min(), osv["month"].max(), freq="MS")
    if len(months) != osv["month"].nunique():
        raise ValueError(f"В файле «{OSV_FILE}» в шапке пропущены месяцы между {months[0]:%Y-%m} и {months[-1]:%Y-%m}.")
    osv = osv.sort_values(["ls", "month"], ignore_index=True)
    osv.attrs["issues"] = issues
    return osv


# ---------------------------------------------------------------------------
# Файл 03 — оплаты
# ---------------------------------------------------------------------------

def read_payments(data_dir: Path) -> pd.DataFrame:
    """Файл 03: ls, payment_date, amount, payment_method (код способа оплаты, «NULL» → <NA>).

    Одинаковые строки не удаляем: ОСВ сходится с суммой вместе с ними, то есть это
    отдельные платежи. Отрицательные суммы (сторно) тоже остаются.
    """
    issues: dict = {}
    raw = pd.read_csv(_require_file(data_dir / PAYMENTS_FILE), sep=";", encoding="utf-8-sig", dtype=str)
    raw.columns = [str(c).strip() for c in raw.columns]
    _require_columns(raw, ["Номер", "Дата оплаты", "Сумма", "Способ оплаты"], PAYMENTS_FILE)

    df = _clean_ls(raw, "Номер", PAYMENTS_FILE, issues)
    df["payment_date"] = pd.to_datetime(df["Дата оплаты"], format="%d.%m.%Y", errors="coerce")
    amount = df["Сумма"].str.replace(r"[\s ]", "", regex=True).str.replace(",", ".", regex=False)
    df["amount"] = pd.to_numeric(amount, errors="coerce").astype("float64")
    invalid = df["payment_date"].isna() | df["amount"].isna()
    if invalid.any():
        issues["строк с неверной датой или суммой"] = int(invalid.sum())
        logger.warning("Файл «%s»: убрано %d строк с неверной датой или суммой.", PAYMENTS_FILE, int(invalid.sum()))
    df = df.loc[~invalid].reset_index(drop=True)

    method_text = df["Способ оплаты"].str.strip().replace({"NULL": None, "": None})
    method = pd.to_numeric(method_text, errors="coerce")
    unknown = method.isna() & method_text.notna()
    if unknown.any():
        issues["нераспознанных способов оплаты"] = int(unknown.sum())
        logger.warning("Файл «%s»: %d способов оплаты не число, записаны как пустые.", PAYMENTS_FILE, int(unknown.sum()))

    payments = pd.DataFrame({
        "ls": df["ls"],
        "payment_date": df["payment_date"].astype("datetime64[ns]"),
        "amount": df["amount"],
        "payment_method": method.astype("Int64"),
    })
    payments.attrs["issues"] = issues
    return payments


# ---------------------------------------------------------------------------
# Файлы 04–13 — меры
# ---------------------------------------------------------------------------

def read_measure_events(data_dir: Path) -> pd.DataFrame:
    """Файлы 04–13: ls, measure, event_date, date_precision. Первая строка файла — название меры, вторая — шапка.

    Повторы строк не удаляем: их число нужно для n_raw в журнале мер (docs/interfaces.md, раздел 3).
    """
    issues: dict = {}
    frames = []
    for measure, file_name in MEASURE_FILES.items():
        raw = pd.read_excel(_require_file(data_dir / file_name), header=1)
        raw.columns = [str(c).strip() for c in raw.columns]
        _require_columns(raw, ["ЛС", "Дата"], file_name)
        file_issues: dict = {}
        df = _clean_ls(raw, "ЛС", file_name, file_issues)
        df["event_date"] = pd.to_datetime(df["Дата"], errors="coerce")
        invalid = df["event_date"].isna()
        if invalid.any():
            file_issues["строк с неверной датой"] = int(invalid.sum())
            logger.warning("Файл «%s»: убрано %d строк с неверной датой.", file_name, int(invalid.sum()))
        df = df.loc[~invalid, ["ls", "event_date"]]

        if measure in MONTH_PRECISION_MEASURES:
            not_first = int((df["event_date"].dt.day != 1).sum())
            if not_first:
                logger.warning(
                    "Файл «%s»: %d дат не на 1-е число, хотя для этой меры известен только месяц.", file_name, not_first,
                )
        repeats = int(df.duplicated().sum())
        if repeats:
            logger.info("Файл «%s»: %d повторов (ЛС, дата), оставлены для подсчёта n_raw.", file_name, repeats)
        df["measure"] = measure
        frames.append(df)
        issues.update({f"{measure}: {key}": value for key, value in file_issues.items()})

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
    events.attrs["issues"] = issues
    return events


# ---------------------------------------------------------------------------
# Файл 14 — лимиты
# ---------------------------------------------------------------------------

def _limits_frame(limits: dict[str, int | None]) -> pd.DataFrame:
    return pd.DataFrame({
        "measure": pd.Categorical(list(MEASURE_FILES), categories=list(MEASURE_FILES)),
        "limit": pd.array([limits[m] for m in MEASURE_FILES], dtype="Int64"),
    })


def _read_limits_yaml(path: Path) -> dict[str, int | None]:
    with open(_require_file(path), encoding="utf-8") as fh:
        limits = (yaml.safe_load(fh) or {}).get("limits") or {}
    missing = [m for m in MEASURE_FILES if m not in limits]
    if missing:
        raise ValueError(f"В файле «{path}» нет лимитов для мер: {', '.join(missing)}.")
    return {m: None if limits[m] is None else int(limits[m]) for m in MEASURE_FILES}


def read_limits(data_dir: Path, settings: dict) -> pd.DataFrame:
    """Файл 14: measure, limit. «нет ограничений» → <NA>.

    Меры сопоставляются по номеру файла в начале названия («04 Автодозвон ХК» → autodial).
    Если файла 14 нет, лимиты берутся из запасного limits.yaml. Расхождения с ним — в лог.
    """
    fallback = REPO_ROOT / settings["paths"]["limits_fallback"]
    path = data_dir / LIMITS_FILE
    if not path.exists():
        logger.warning("Не найден файл «%s», лимиты взяты из «%s».", LIMITS_FILE, fallback)
        return _limits_frame(_read_limits_yaml(fallback))

    raw = pd.read_excel(path, header=None, dtype=str)
    by_number = {name[:2]: measure for measure, name in MEASURE_FILES.items()}
    limits: dict[str, int | None] = {}
    for name, value in raw.iloc[1:, :2].itertuples(index=False):
        if pd.isna(name) or not str(name).strip():
            continue
        found = re.match(r"\s*(\d{2})\b", str(name))
        measure = by_number.get(found.group(1)) if found else None
        if measure is None:
            raise ValueError(f"В файле «{LIMITS_FILE}» неизвестная мера «{name}».")
        text = "" if pd.isna(value) else str(value).strip()
        if text.lower() == NO_LIMIT_TEXT:
            limits[measure] = None
        elif re.fullmatch(r"\d+(\.0+)?", text):
            limits[measure] = int(float(text))
        else:
            raise ValueError(
                f"В файле «{LIMITS_FILE}» для меры «{name}» лимит «{text}»: нужно целое число или «{NO_LIMIT_TEXT}»."
            )
    missing = [MEASURE_FILES[m] for m in MEASURE_FILES if m not in limits]
    if missing:
        raise ValueError(f"В файле «{LIMITS_FILE}» нет лимитов для мер: {', '.join(missing)}.")

    if fallback.exists():
        reserve = _read_limits_yaml(fallback)
        differ = [f"{m}: {limits[m]} против {reserve[m]}" for m in MEASURE_FILES if limits[m] != reserve[m]]
        if differ:
            logger.warning(
                "Лимиты в файле 14 отличаются от «%s» (используется файл 14): %s.", fallback.name, "; ".join(differ),
            )
    return _limits_frame(limits)


# ---------------------------------------------------------------------------
# Проверка типов и кэш
# ---------------------------------------------------------------------------

def check_schema(name: str, df: pd.DataFrame) -> None:
    """Проверяет колонки и типы таблицы по SCHEMAS. Ошибка — если что-то не совпадает."""
    problems = []
    for column, expected in SCHEMAS[name].items():
        if column not in df.columns:
            problems.append(f"нет колонки «{column}»")
            continue
        dtype = df[column].dtype
        if isinstance(expected, list):
            if not isinstance(dtype, pd.CategoricalDtype) or list(dtype.categories) != expected:
                problems.append(f"«{column}»: ожидалась category {expected}, получено {dtype}")
        elif str(dtype) != expected:
            problems.append(f"«{column}»: ожидался тип {expected}, получено {dtype}")
    extra = [c for c in df.columns if c not in SCHEMAS[name]]
    if extra:
        problems.append(f"лишние колонки: {', '.join(extra)}")
    if "ls" in df.columns and df["ls"].isna().any():
        problems.append("есть пустые ЛС")
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
        return _read_table(name, data_dir, settings)

    key = _cache_key(name, data_dir, settings)
    table_path, meta_path = cache_dir / f"{name}.parquet", cache_dir / f"{name}.json"
    if table_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("key") == key:
                table = pd.read_parquet(table_path)
                table.attrs = {"issues": meta.get("issues", {}), "cached": True}
                logger.info("Таблица «%s» взята из кэша.", name)
                return table
        except (OSError, ValueError) as error:
            logger.warning("Кэш таблицы «%s» не прочитан (%s), читаю исходные файлы.", name, error)

    table = _read_table(name, data_dir, settings)
    cache_dir.mkdir(parents=True, exist_ok=True)
    issues = table.attrs.get("issues", {})
    table.attrs = {}
    table.to_parquet(table_path, index=False)
    meta_path.write_text(json.dumps({"key": key, "issues": issues}, ensure_ascii=False, indent=1), encoding="utf-8")
    table.attrs = {"issues": issues, "cached": False}
    return table


def load_all(data_dir: Path | str, settings: dict, cache_dir: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Читает файлы 01–14 и возвращает таблицы info, osv, payments, events, limits.

    cache_dir — папка кэша для таблиц 01–13; None — читать исходные файлы без кэша.
    Каждая таблица проходит check_schema. Проблемы разбора — в table.attrs["issues"].
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Не найдена папка с данными: {data_dir}")
    cache_dir = Path(cache_dir) if cache_dir is not None else None

    tables = {}
    for name in TABLE_SOURCES:
        logger.info("Читаю «%s».", name)
        tables[name] = _load_cached(name, data_dir, settings, cache_dir)
    tables["limits"] = read_limits(data_dir, settings)
    for name, table in tables.items():
        check_schema(name, table)
    return tables


def default_cache_dir(settings: dict) -> Path:
    return REPO_ROOT / settings["paths"]["cache_dir"]


# ---------------------------------------------------------------------------
# Командная строка: сводка по таблицам
# ---------------------------------------------------------------------------

def summarize(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """По каждой таблице: строк, ЛС, мин./макс. дата, проблемы разбора."""
    rows = []
    for name, table in tables.items():
        dates = [c for c in table.columns if pd.api.types.is_datetime64_any_dtype(table[c])]
        issues = table.attrs.get("issues", {})
        rows.append({
            "таблица": name,
            "строк": len(table),
            "ЛС": table["ls"].nunique() if "ls" in table.columns else None,
            "с": f"{table[dates[0]].min():%Y-%m-%d}" if dates else "",
            "по": f"{table[dates[0]].max():%Y-%m-%d}" if dates else "",
            "проблемы разбора": "; ".join(f"{k}: {v}" for k, v in issues.items()) or "нет",
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Читает файлы 01–14 и печатает сводку по таблицам.")
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
        logger.error("Файлы не прочитаны: %s", error)
        return 1

    with pd.option_context("display.max_colwidth", None, "display.width", 200):
        print(summarize(tables).to_string(index=False))
    print()
    print(tables["limits"].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
