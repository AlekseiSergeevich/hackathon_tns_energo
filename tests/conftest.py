"""Маленькие входные файлы 01–14 в том же виде, что и настоящие, для тестов загрузчиков."""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import openpyxl
import pandas as pd
import pytest
import yaml

from tns.data import loaders

REAL_DATA_DIR = loaders.REPO_ROOT / "Материалы к кейсу"

INFO_HEADER = [
    "ЛС", "Возможность дистанционного отключения", "Наличие телефона", "Наличие льгот", "Газификация дома",
    "Город", "ЯрОблИЕРЦ квитанция", "Почта России квитанция", "электронная квитанция", "не проживает",
    "ЧД", "МКД", "Общежитие", "Установка Тамбур", "Установка опора", "Установка в квартире/доме",
    "Установка лестничкая клетка", "Адрес (ГУИД)",
]
OSV_METRICS = ["СЗ на начало", "Начислено", "Опалачено"]
LIMITS_YAML = {
    "limits": {
        "autodial": 8000, "email": None, "sms": 2150, "operator_call": 1550, "claim": 400, "visit": 500,
        "restriction_notice": 6200, "restriction": 200, "court_application": 400, "court_result": 250,
    }
}


def info_row(ls, house="МКД", **flags) -> list:
    """Строка файла 01. Флаги по умолчанию «Нет»; flags задаются по именам колонок с _ вместо пробелов."""
    row = dict.fromkeys(INFO_HEADER, "Нет")
    row.update({"ЛС": ls, "Адрес (ГУИД)": f"guid-{ls}"})
    if house:
        row[house] = "Да"
    row.update({key.replace("_", " "): value for key, value in flags.items()})
    return [row[column] for column in INFO_HEADER]


def write_xlsx(path: Path, rows: list[list]) -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    for row in rows:
        ws.append(row)
    wb.save(path)
    return path


def write_info(path: Path, rows: list[list] | None = None) -> Path:
    default = [
        info_row(1, Наличие_телефона="Да", электронная_квитанция="Да"),
        info_row(2, house="ЧД", Наличие_льгот="Да", Возможность_дистанционного_отключения="Да"),
        info_row(3, house=None, Газификация_дома="Да"),
    ]
    return write_xlsx(path, [INFO_HEADER, *(rows if rows is not None else default)])


def write_osv(path: Path, blocks: list[tuple[datetime, list[list]]], ls_values: list, metrics=OSV_METRICS) -> Path:
    """Файл 02: двухуровневая шапка (месяц над тремя показателями), месяц указан только в первой колонке блока."""
    top, sub = ["ЛС"], [None]
    for month, _ in blocks:
        top += [month, None, None]
        sub += list(metrics)
    rows = [top, sub]
    for i, ls in enumerate(ls_values):
        rows.append([ls] + [value for _, values in blocks for value in values[i]])
    return write_xlsx(path, rows)


def default_osv_blocks() -> tuple[list[tuple[datetime, list[list]]], list]:
    """Три ЛС; февраль повторён в шапке, второй блок продолжает первый (сальдо + начислено − оплачено)."""
    jan = [[0, 100, 100], [500, 100, 0], [-50, 100, 100]]
    feb = [[0, 100, 50], [600, 100, 100], [-50, 100, 0]]
    feb_again = [[50, 100, 100], [600, 100, 0], [50, 100, 100]]
    return (
        [(datetime(2025, 1, 1), jan), (datetime(2025, 2, 1), feb), (datetime(2025, 2, 1), feb_again)],
        [10, 20, 30],
    )


def write_payments(path: Path, lines: list[str] | None = None) -> Path:
    body = lines if lines is not None else [
        "10;15.01.2025;100;5",
        "20;05.02.2025;1 240,50;NULL",
        "20;05.02.2025;1 240,50;NULL",
        "30;07.02.2025;-30,5;0",
    ]
    path.write_text("\n".join(["Номер;Дата оплаты;Сумма;Способ оплаты", *body]) + "\n", encoding="utf-8-sig")
    return path


def write_measure(path: Path, title: str, rows: list[list]) -> Path:
    """Файлы 04–13: первая строка — название меры, вторая — шапка «ЛС | Дата»."""
    return write_xlsx(path, [[title, None], ["ЛС", "Дата"], *rows])


def write_limits(path: Path, rows: list[list] | None = None) -> Path:
    default = [
        [f"{name[:2]} {name[3:].removesuffix('.xlsx')}", value]
        for name, value in zip(
            loaders.MEASURE_FILES.values(),
            [8000, "нет ограничений", 2150, 1550, 400, 500, 6200, 200, 400, 250],
        )
    ]
    return write_xlsx(path, [["Наименование меры воздействия", "Лимит в месяц"], *(rows if rows is not None else default)])


@pytest.fixture
def settings(tmp_path) -> dict:
    """Настройки репозитория, но с запасными лимитами во временной папке."""
    data = loaders.load_settings()
    fallback = tmp_path / "limits.yaml"
    fallback.write_text(yaml.safe_dump(LIMITS_YAML), encoding="utf-8")
    data["paths"] = {**data["paths"], "limits_fallback": str(fallback)}
    data["osv"] = {**data["osv"], "duplicate_month": "shift_next"}
    return data


@pytest.fixture
def data_dir(tmp_path) -> Path:
    """Папка с файлами 01–14 из нескольких строк. Файлы можно переписывать в тестах."""
    folder = tmp_path / "data"
    folder.mkdir()
    write_info(folder / loaders.INFO_FILE)
    blocks, ls_values = default_osv_blocks()
    write_osv(folder / loaders.OSV_FILE, blocks, ls_values)
    write_payments(folder / loaders.PAYMENTS_FILE)
    for measure, file_name in loaders.MEASURE_FILES.items():
        write_measure(
            folder / file_name,
            measure,
            [[10, datetime(2025, 1, 1 if measure in loaders.MONTH_PRECISION_MEASURES else 26)]],
        )
    write_limits(folder / loaders.LIMITS_FILE)
    return folder


@pytest.fixture(scope="session")
def real_data_dir() -> Path:
    if not (REAL_DATA_DIR / loaders.OSV_FILE).exists():
        pytest.skip("Нет папки «Материалы к кейсу» с настоящими данными.")
    return REAL_DATA_DIR


def copy_without(folder: Path, file_name: str, target: Path) -> Path:
    shutil.copytree(folder, target)
    (target / file_name).unlink()
    return target


# ---------------------------------------------------------------------------
# Таблицы load_all, собранные вручную (для тестов очистки и панели)
# ---------------------------------------------------------------------------

def make_info(ls_values) -> pd.DataFrame:
    n = len(ls_values)
    info = pd.DataFrame({"ls": pd.Series(ls_values, dtype="int64")})
    for flag in loaders.INFO_FLAGS:
        info[flag] = pd.array([False] * n, dtype="boolean")
    info["house_type"] = pd.Categorical(["МКД"] * n, categories=loaders.HOUSE_TYPES)
    info["address_guid"] = pd.array([f"g{ls}" for ls in ls_values], dtype="string")
    return info


def make_osv(rows: dict[int, list[tuple[float, float, float]]]) -> pd.DataFrame:
    """{ЛС: [(сальдо, начислено, оплачено) по месяцам с 2025-01]}."""
    records = [
        {"ls": ls, "month": pd.Timestamp("2025-01-01") + pd.DateOffset(months=i), "saldo_start": s, "accrual": a, "paid": p}
        for ls, months in rows.items() for i, (s, a, p) in enumerate(months)
    ]
    return pd.DataFrame(records).astype({
        "ls": "int64", "month": "datetime64[ns]", "saldo_start": "float64", "accrual": "float64", "paid": "float64",
    })


def make_payments(rows) -> pd.DataFrame:
    """[(ЛС, дата, сумма, способ)]."""
    frame = pd.DataFrame(rows, columns=["ls", "payment_date", "amount", "payment_method"])
    return frame.astype({"ls": "int64", "payment_date": "datetime64[ns]", "amount": "float64", "payment_method": "Int64"})


def make_events(rows) -> pd.DataFrame:
    """[(ЛС, мера, дата)]."""
    frame = pd.DataFrame(rows, columns=["ls", "measure", "event_date"]).astype({"ls": "int64", "event_date": "datetime64[ns]"})
    frame["measure"] = pd.Categorical(frame["measure"], categories=list(loaders.MEASURE_FILES))
    frame["date_precision"] = pd.Categorical(
        ["month" if m in loaders.MONTH_PRECISION_MEASURES else "day" for m in frame["measure"]], categories=["month", "day"]
    )
    return frame
