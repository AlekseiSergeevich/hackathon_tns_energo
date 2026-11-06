"""Сверка панели с исходными файлами (DATA-8): ОСВ (02) и оплаты (03).

Исходные файлы читаются здесь заново и по-простому (openpyxl и csv), без tns.data.loaders,
чтобы ошибка загрузчика не повторилась в проверке. Сверяются все ЛС:

1. ОСВ → панель: сальдо, начислено, оплачено в каждой ячейке совпадают с панелью; dz = max(0, сальдо);
   каждый ЛС из ОСВ либо в панели, либо объяснено, почему его нет.
2. Файл 03 → панель: paid_payments = сумма оплат ЛС за месяц.
3. «Оплачено» ОСВ ↔ оплаты 03: расхождения раскладываются по причинам (перенос на соседний месяц,
   оплаты декабря 2024, конец данных, сальдо подтверждает файл 03). Остальное — «не объяснено».

Запуск: python -m tns.data.panel_check --month 2026-03 [--data <папка>] [--ls 10 20 ...]
Читает output/<ГГГГ-ММ>/panel.parquet и пишет туда же panel_check.xlsx: итог по всем ЛС,
20 ЛС для ручной проверки с адресами ячеек файла 02 и список необъяснённых расхождений.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.utils import get_column_letter

from tns.data.loaders import OSV_FILE, PAYMENTS_FILE, REPO_ROOT, load_settings
from tns.data.panel import parse_plan_month

logger = logging.getLogger(__name__)

TOLERANCE = 0.005  # меньше полкопейки — погрешность float
METRIC_LABELS = {"saldo_start": "сз на начало", "accrual": "начислено", "paid": ("оплачено", "опалачено")}

# Причины расхождения «Оплачено» ОСВ и оплат 03.
SHIFT = "перенос на соседний месяц"
BEFORE_DATA = "оплаты декабря 2024 проведены в ОСВ январём"
CONFIRMED_BY_SALDO = "«Оплачено» ОСВ не подтверждается сальдо: сальдо следующего месяца сходится с файлом 03"
END_OF_DATA = "конец данных: оплаты последнего месяца ещё не проведены в ОСВ"
UNEXPLAINED = "не объяснено"
REASON_TEXT = {
    SHIFT: "Оплата в конце месяца по файлу 03 проведена в ОСВ соседним месяцем: за два месяца суммы равны.",
    BEFORE_DATA: "В файле 03 оплата датирована декабрём 2024 (вне панели), в ОСВ проведена в январе 2025.",
    CONFIRMED_BY_SALDO: "Сальдо ОСВ на начало следующего месяца = сальдо + начислено − оплаты по файлу 03. "
                        "Ошибка в колонке «Оплачено» ОСВ, сальдо верное. В расчётах панели «Оплачено» ОСВ не используется.",
    END_OF_DATA: "Оплата по файлу 03 есть, в ОСВ её ещё нет: она будет проведена следующим месяцем, которого нет в данных.",
    UNEXPLAINED: "Причину по данным установить не удалось.",
}


# ---------------------------------------------------------------------------
# Независимое чтение исходных файлов
# ---------------------------------------------------------------------------

def read_osv_cells(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Все ячейки ОСВ: ls, month, row, saldo_start/accrual/paid и их адреса (*_cell).

    Месяцы блоков идут подряд от первого месяца шапки: k-й блок — первый месяц + k.
    Если подпись в шапке с этим не совпадает (как повтор «2026-02»), это записывается в notes.
    """
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook.worksheets[0]
    rows = sheet.iter_rows(values_only=True)
    top, sub = next(rows), next(rows)

    starts = [i for i, value in enumerate(top) if isinstance(value, datetime)]
    if not starts:
        raise ValueError(f"В файле «{path.name}» в первой строке шапки нет месяцев.")
    first = pd.Timestamp(top[starts[0]]).to_period("M").to_timestamp()
    blocks, notes = [], []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(top)
        month = first + pd.DateOffset(months=k)
        label = pd.Timestamp(top[start]).to_period("M").to_timestamp()
        if label != month:
            notes.append(f"Блок {k + 1} (колонка {get_column_letter(start + 1)}) подписан {label:%Y-%m}, "
                         f"по порядку блоков это {month:%Y-%m}.")
        columns = {}
        for metric, labels in METRIC_LABELS.items():
            labels = (labels,) if isinstance(labels, str) else labels
            found = [i for i in range(start, end) if str(sub[i] or "").strip().lower() in labels]
            if len(found) != 1:
                raise ValueError(f"В файле «{path.name}» у блока {month:%Y-%m} не найдена колонка «{labels[0]}».")
            columns[metric] = found[0]
        blocks.append((month, columns))

    records = []
    for row_number, row in enumerate(rows, start=3):
        ls = row[0]
        if isinstance(ls, bool) or not isinstance(ls, (int, float)) or ls != int(ls):
            records.append({"ls": None, "row": row_number, "raw_ls": ls})
            continue
        for month, columns in blocks:
            record = {"ls": int(ls), "row": row_number, "month": month}
            for metric, index in columns.items():
                value = row[index] if index < len(row) else None
                record[metric] = float(value) if isinstance(value, (int, float)) else np.nan
                record[f"{metric}_cell"] = f"{get_column_letter(index + 1)}{row_number}"
            records.append(record)
    workbook.close()
    cells = pd.DataFrame(records)
    if "raw_ls" not in cells.columns:
        cells["raw_ls"] = None  # строк без номера ЛС нет
    return cells, notes


def read_payment_lines(path: Path) -> pd.DataFrame:
    """Оплаты файла 03: ls, payment_date, amount. Строки, которые не читаются, пропускаются."""
    records = []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        next(reader)
        for line in reader:
            try:
                ls = int(line[0].strip())
                date = datetime.strptime(line[1].strip(), "%d.%m.%Y")
                amount = float(line[2].replace(" ", "").replace(" ", "").replace(",", "."))
            except (ValueError, IndexError):
                continue
            records.append((ls, date, amount))
    payments = pd.DataFrame(records, columns=["ls", "payment_date", "amount"])
    payments["month"] = payments["payment_date"].dt.to_period("M").dt.to_timestamp()
    return payments


# ---------------------------------------------------------------------------
# Сверка
# ---------------------------------------------------------------------------

def _differs(a: pd.Series, b: pd.Series) -> pd.Series:
    """Значения различаются: больше полкопейки или одно пустое, а другое нет."""
    both_nan = a.isna() & b.isna()
    return ~both_nan & ~((a - b).abs() <= TOLERANCE)


def compare(panel: pd.DataFrame, cells: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    """Строка на каждую пару ЛС × месяц из ОСВ: значения из файлов, из панели и итог сверки."""
    osv = cells[cells["ls"].notna()].astype({"ls": "int64"})
    columns = ["ls", "month", "saldo_start", "dz", "accrual", "paid", "paid_payments"]
    table = osv.merge(panel[columns], on=["ls", "month"], how="left", suffixes=("_osv", ""), indicator=True)
    table["in_panel"] = table.pop("_merge") == "both"

    monthly = payments.groupby(["ls", "month"], as_index=False).agg(pay_sum=("amount", "sum"), pay_count=("amount", "size"))
    table = table.merge(monthly, on=["ls", "month"], how="left")
    table["pay_count"] = table["pay_count"].fillna(0).astype("int64")
    table["pay_sum"] = table["pay_sum"].fillna(0.0)

    inside = table["in_panel"]
    table["osv_ok"] = ~inside | ~(
        _differs(table["saldo_start_osv"], table["saldo_start"])
        | _differs(table["accrual_osv"], table["accrual"])
        | _differs(table["paid_osv"], table["paid"])
        | _differs(table["saldo_start_osv"].clip(lower=0), table["dz"])
    )
    table["payments_ok"] = ~inside | ~_differs(table["pay_sum"], table["paid_payments"])
    table["paid_reason"] = _paid_reasons(table, payments)
    return table.sort_values(["ls", "month"], ignore_index=True)


def _shifts(diff: pd.Series, ls: pd.Series, max_months: int) -> pd.Series:
    """Строки, где разница — перенос оплаты между соседними месяцами.

    Эпизод начинается с месяца с разницей. Если не позже чем через max_months − 1 месяцев
    накопленная разница возвращается к уровню до эпизода, все месяцы эпизода — перенос:
    за эти месяцы ОСВ и файл 03 в сумме совпадают.
    """
    result = pd.Series(False, index=diff.index)
    has_diff = (diff.abs() > 0.01).groupby(ls).transform("any")
    for _, part in diff[has_diff].groupby(ls[has_diff]):
        values = part.to_numpy()
        cum = np.round(np.cumsum(values), 2)
        i = 0
        while i < len(values):
            if abs(values[i]) <= 0.01:
                i += 1
                continue
            base = cum[i - 1] if i else 0.0
            end = next((j for j in range(i + 1, min(i + max_months, len(values))) if abs(cum[j] - base) <= 0.01), None)
            if end is None:
                i += 1
                continue
            result[part.index[i:end + 1]] = True
            i = end + 1
    return result


SHIFT_MONTHS = 3  # перенос оплаты укладывается не больше чем в 3 соседних месяца


def _paid_reasons(table: pd.DataFrame, payments: pd.DataFrame) -> pd.Series:
    """Причина расхождения «Оплачено» ОСВ и оплат 03 в строке; пусто, если суммы равны."""
    t = table.sort_values(["ls", "month"])
    diff = (t["pay_sum"] - t["paid_osv"].fillna(0)).round(2)
    next_diff = diff.groupby(t["ls"]).shift(-1).fillna(0.0)
    first, last = t["month"].min(), t["month"].max()
    before = payments[payments["month"] < first].groupby("ls")["amount"].sum()
    december = t["ls"].map(before).fillna(0.0)
    saldo_next = t.groupby("ls")["saldo_start_osv"].shift(-1)
    # Сальдо следующего месяца сходится с оплатами 03. С «Оплачено» ОСВ оно тогда сойтись не может: суммы разные.
    by_payments = (saldo_next - (t["saldo_start_osv"] + t["accrual_osv"] - t["pay_sum"])).abs() <= 0.01

    reason = pd.Series(None, index=t.index, dtype="object")
    differs = diff.abs() > 0.01
    rules = [
        (SHIFT, _shifts(diff, t["ls"], SHIFT_MONTHS)),
        (BEFORE_DATA, (t["month"] == first) & (diff < 0)
         & (((diff + december).abs() <= 0.01) | ((diff + next_diff + december).abs() <= 0.01))),
        (CONFIRMED_BY_SALDO, by_payments),
        (END_OF_DATA, (t["month"] == last) & (diff > 0)),
    ]
    for name, mask in rules:
        reason[differs & reason.isna() & mask] = name
    reason[differs & reason.isna()] = UNEXPLAINED
    return reason.reindex(table.index)


def summary(table: pd.DataFrame, cells: pd.DataFrame, panel: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """Итог сверки по всем ЛС: проверка, сколько проверено, сколько расхождений и чем объяснены."""
    rows = []
    bad_rows = cells[cells["ls"].isna()]
    osv_ls = table["ls"].unique()
    missing = np.setdiff1d(osv_ls, panel["ls"].unique())
    zero = (table[["saldo_start_osv", "accrual_osv", "paid_osv"]].fillna(0).abs() < TOLERANCE).all(axis=1)
    inactive = int(zero.groupby(table["ls"]).all().reindex(missing).sum())
    extra = np.setdiff1d(panel["ls"].unique(), osv_ls)
    rows.append(("ЛС из ОСВ есть в панели", len(osv_ls), len(osv_ls) - len(missing), len(missing), inactive,
                 f"нет в панели: {len(missing)} ЛС, из них без движения (все суммы нулевые, убраны очисткой Q-13): {inactive}"))
    rows.append(("Строки ОСВ без номера ЛС", len(cells["row"].unique()), len(cells["row"].unique()) - len(bad_rows), len(bad_rows),
                 len(bad_rows), "убраны загрузчиком: " + ", ".join(f"строка {r} «{v}»" for r, v in zip(bad_rows["row"], bad_rows["raw_ls"]))))
    rows.append(("ЛС панели есть в ОСВ", panel["ls"].nunique(), panel["ls"].nunique() - len(extra), len(extra), 0, ""))
    inside = table[table["in_panel"]]
    rows.append(("Сальдо, начислено, оплачено, dz: ОСВ = панель", len(inside), int(inside["osv_ok"].sum()),
                 int((~inside["osv_ok"]).sum()), 0, "сравнение каждой ячейки файла 02 с панелью"))
    rows.append(("paid_payments = сумма оплат файла 03", len(inside), int(inside["payments_ok"].sum()),
                 int((~inside["payments_ok"]).sum()), 0, "сумма оплат ЛС за месяц по файлу 03"))
    reasons = inside["paid_reason"].value_counts()
    differs = int(reasons.sum())
    rows.append(("«Оплачено» ОСВ = оплаты файла 03", len(inside), len(inside) - differs, differs,
                 differs - int(reasons.get(UNEXPLAINED, 0)), "причины — ниже"))
    for reason in [SHIFT, BEFORE_DATA, CONFIRMED_BY_SALDO, END_OF_DATA, UNEXPLAINED]:
        count = int(reasons.get(reason, 0))
        rows.append((f"   — {reason}", None, None, count, 0 if reason == UNEXPLAINED else count, REASON_TEXT[reason]))
    for note in notes:
        rows.append(("Шапка ОСВ", None, None, None, None, note + " Панель строится так же (osv.duplicate_month: shift_next)."))
    return pd.DataFrame(rows, columns=["Проверка", "Проверено", "Совпало", "Расхождений", "Из них объяснено", "Пояснение"])


# ---------------------------------------------------------------------------
# 20 ЛС для ручной проверки
# ---------------------------------------------------------------------------

def pick_cases(table: pd.DataFrame, count: int = 20, seed: int = 2026) -> dict[str, int]:
    """Половина — ЛС с разными ситуациями (наименьший номер с ней), половина — случайные ЛС панели."""
    t = table[table["in_panel"]]
    by_ls = t.groupby("ls")
    no_issue = by_ls["paid_reason"].apply(lambda r: r.isna().all())
    situations = {
        "долг каждый месяц, всё сходится": (by_ls["dz"].min() > 0) & no_issue,
        "переплата: отрицательное сальдо, dz = 0": (by_ls["saldo_start"].min() < -100) & no_issue,
        "отрицательное начисление (перерасчёт)": (by_ls["accrual"].min() < -100),
        "крупный долг": by_ls["dz"].max() >= by_ls["dz"].max().quantile(0.999),
    }
    situations.update({
        f"«Оплачено» ОСВ ≠ 03: {reason}": t[t["paid_reason"] == reason].groupby("ls").size() > 0
        for reason in [SHIFT, BEFORE_DATA, CONFIRMED_BY_SALDO, END_OF_DATA, UNEXPLAINED]
    })
    situations["нет в панели: ЛС без движения"] = table.loc[~table["in_panel"]].groupby("ls").size() > 0
    cases, used = {}, set()
    for name, mask in situations.items():
        candidates = [int(x) for x in mask[mask].index if int(x) not in used]
        if candidates:
            cases[name] = min(candidates)
            used.add(cases[name])
    pool = np.setdiff1d(t["ls"].unique(), list(used))
    rng = np.random.default_rng(seed)
    for ls in sorted(rng.choice(pool, size=min(count - len(cases), len(pool)), replace=False)):
        cases[f"случайный ЛС {ls}"] = int(ls)
    return dict(list(cases.items())[:count])


def manual_table(table: pd.DataFrame, cases: dict[str, int]) -> pd.DataFrame:
    rows = []
    for situation, ls in cases.items():
        for _, r in table[table["ls"] == ls].iterrows():
            rows.append({
                "Ситуация": situation,
                "ЛС": ls,
                "Строка в файле 02": r["row"],
                "Месяц": f"{r['month']:%Y-%m}",
                "Ячейка «СЗ на начало»": r["saldo_start_cell"],
                "Сальдо (02)": r["saldo_start_osv"],
                "Сальдо (панель)": r["saldo_start"],
                "dz (панель)": r["dz"],
                "Ячейка «Начислено»": r["accrual_cell"],
                "Начислено (02)": r["accrual_osv"],
                "Начислено (панель)": r["accrual"],
                "Ячейка «Оплачено»": r["paid_cell"],
                "Оплачено (02)": r["paid_osv"],
                "Оплачено (панель)": r["paid"],
                "Оплат в 03, шт.": r["pay_count"],
                "Оплаты 03, руб.": r["pay_sum"],
                "paid_payments (панель)": r["paid_payments"],
                "02 = панель": "нет в панели" if not r["in_panel"] else ("да" if r["osv_ok"] else "НЕТ"),
                "03 = панель": "нет в панели" if not r["in_panel"] else ("да" if r["payments_ok"] else "НЕТ"),
                "«Оплачено» 02 ≠ 03: причина": r["paid_reason"] or "",
            })
    return pd.DataFrame(rows)


HOW_TO = [
    "Сверка панели с ОСВ (DATA-8)",
    "",
    "Лист «Итог» — проверка всех ЛС: каждая ячейка файла 02 и каждая оплата файла 03 сравниваются с панелью.",
    "Лист «20 ЛС» — для ручной проверки. По каждому ЛС: номер строки и адреса ячеек в файле 02.",
    "  1. Откройте «02 Обортно-сальдовая ведомость ЛС ХК.xlsx», перейдите в ячейку из колонки «Ячейка …» (Ctrl+G в Excel).",
    "  2. Сравните число в ячейке с колонками «(02)» и «(панель)».",
    "  3. Для оплат: отфильтруйте «03 Оплаты ХК.csv» по номеру ЛС и месяцу и сложите суммы — это «Оплаты 03, руб.».",
    "Ожидается: сальдо, начислено и оплачено совпадают с панелью; dz = сальдо, если оно больше 0, иначе 0.",
    "«Оплачено» ОСВ и оплаты 03 могут различаться — причина в последней колонке, пояснения на листе «Итог».",
    "Лист «Не объяснено» — строки, где причину расхождения установить не удалось.",
]


def write_check(path: Path, summary_table: pd.DataFrame, manual: pd.DataFrame, unexplained: pd.DataFrame) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"": HOW_TO}).to_excel(writer, sheet_name="Как проверить", index=False, header=False)
        summary_table.to_excel(writer, sheet_name="Итог", index=False)
        manual.to_excel(writer, sheet_name="20 ЛС", index=False)
        unexplained.to_excel(writer, sheet_name="Не объяснено", index=False)
        writer.sheets["Как проверить"].column_dimensions["A"].width = 130
        for column, width in zip("ABCDEF", [52, 12, 12, 12, 16, 120]):
            writer.sheets["Итог"].column_dimensions[column].width = width
        sheet = writer.sheets["20 ЛС"]
        for index in range(1, len(manual.columns) + 1):
            sheet.column_dimensions[get_column_letter(index)].width = 14
        sheet.column_dimensions["A"].width = 44
        sheet.column_dimensions[get_column_letter(len(manual.columns))].width = 60
        sheet.freeze_panes = "E2"


def run_check(panel: pd.DataFrame, data_dir: Path, ls: list[int] | None = None
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Таблица сверки по всем ЛС, итог, 20 ЛС для ручной проверки и необъяснённые строки."""
    logger.info("Читаю «%s» напрямую.", OSV_FILE)
    cells, notes = read_osv_cells(Path(data_dir) / OSV_FILE)
    payments = read_payment_lines(Path(data_dir) / PAYMENTS_FILE)
    table = compare(panel, cells, payments)
    if ls:
        unknown = sorted(set(ls) - set(table["ls"]))
        if unknown:
            raise ValueError(f"В ОСВ нет ЛС: {', '.join(map(str, unknown))}.")
        cases = {f"ЛС {x}": x for x in ls}
    else:
        cases = pick_cases(table)
    columns = ["ls", "month", "row", "saldo_start_osv", "saldo_start", "accrual_osv", "accrual", "paid_osv", "paid",
               "pay_sum", "paid_payments", "osv_ok", "payments_ok", "paid_reason"]
    bad = table[table["in_panel"] & (~table["osv_ok"] | ~table["payments_ok"] | (table["paid_reason"] == UNEXPLAINED))]
    return table, summary(table, cells, panel, notes), manual_table(table, cases), bad[columns]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сверка панели с ОСВ и оплатами: все ЛС + 20 ЛС для ручной проверки.")
    parser.add_argument("--config", help="файл настроек (по умолчанию configs/settings.yaml)")
    parser.add_argument("--data", help="папка с файлами 01–14 (по умолчанию из настроек)")
    parser.add_argument("--out", help="папка с результатами (по умолчанию из настроек)")
    parser.add_argument("--month", required=True, help="месяц плана ГГГГ-ММ: папка, где лежит panel.parquet")
    parser.add_argument("--ls", nargs="+", type=int, help="номера ЛС для ручной проверки (по умолчанию 20 ЛС)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        settings = load_settings(args.config)
        folder = Path(args.out or REPO_ROOT / settings["paths"]["output_dir"]) / f"{parse_plan_month(args.month):%Y-%m}"
        panel_path = folder / "panel.parquet"
        if not panel_path.exists():
            raise FileNotFoundError(f"Нет файла «{panel_path}». Сначала постройте панель: python -m tns.data.panel --month {args.month}")
        data_dir = Path(args.data or REPO_ROOT / settings["paths"]["data_dir"])
        _, summary_table, manual, bad = run_check(pd.read_parquet(panel_path), data_dir, args.ls)
        write_check(folder / "panel_check.xlsx", summary_table, manual, bad)
    except (OSError, ValueError, KeyError) as error:
        logger.error("Сверка не построена: %s", error)
        return 1

    for _, row in summary_table.iterrows():
        if pd.notna(row["Расхождений"]):
            logger.info("%s: расхождений %s, объяснено %s.", row["Проверка"].strip(), row["Расхождений"], row["Из них объяснено"])
    logger.info("Не объяснено строк: %d. Результат записан в «%s».", len(bad), folder / "panel_check.xlsx")
    return 0 if bad.empty else 2


if __name__ == "__main__":
    sys.exit(main())
