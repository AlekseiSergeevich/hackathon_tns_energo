"""Сверка панели с ОСВ и оплатами (DATA-8, tns/data/panel_check.py).

1. Независимое чтение файлов 02 и 03 и сравнение с панелью: совпадение и поиск подстроенных ошибок.
2. Причины расхождений «Оплачено» ОСВ и оплат 03 — на примерах, посчитанных вручную.
3. Команда и проверка на настоящих данных.
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import pytest

from tns.data import loaders, panel, panel_check as pc, quality
from tests.conftest import write_osv, write_xlsx

T = pd.Timestamp


@pytest.fixture
def built(data_dir, settings):
    """Панель из мини-файлов conftest и таблица сверки."""
    tables, _ = panel.build_tables(data_dir, settings)
    return tables["panel"], data_dir


# ---------------------------------------------------------------------------
# Чтение ОСВ напрямую
# ---------------------------------------------------------------------------

class TestReadOsv:
    def test_cells_and_addresses(self, data_dir):
        cells, notes = pc.read_osv_cells(data_dir / loaders.OSV_FILE)
        # ЛС 20 — вторая строка данных (строка 4 в Excel), февраль — второй блок (колонки E, F, G).
        feb = cells[(cells["ls"] == 20) & (cells["month"] == T("2025-02-01"))].iloc[0]
        assert (feb["saldo_start"], feb["accrual"], feb["paid"]) == (600, 100, 100)
        assert (feb["saldo_start_cell"], feb["accrual_cell"], feb["paid_cell"]) == ("E4", "F4", "G4")
        # Третий блок подписан «2025-02», по порядку это март.
        assert cells["month"].drop_duplicates().tolist() == [T("2025-01-01"), T("2025-02-01"), T("2025-03-01")]
        assert notes == ["Блок 3 (колонка H) подписан 2025-02, по порядку блоков это 2025-03."]

    def test_typo_in_header_and_rows_without_ls(self, tmp_path):
        path = write_osv(tmp_path / "02.xlsx", [(datetime(2025, 1, 1), [[1, 2, 3], [4, 5, 6]])], [7, "#N/A"],
                         metrics=["СЗ на начало", "Начислено", "Опалачено"])
        cells, _ = pc.read_osv_cells(path)
        assert cells.loc[cells["ls"] == 7, "paid"].tolist() == [3]
        assert cells.loc[cells["ls"].isna(), ["row", "raw_ls"]].values.tolist() == [[4, "#N/A"]]

    def test_missing_metric_column(self, tmp_path):
        path = write_xlsx(tmp_path / "02.xlsx", [["ЛС", datetime(2025, 1, 1), None], [None, "СЗ на начало", "Начислено"], [1, 0, 0]])
        with pytest.raises(ValueError, match="у блока 2025-01 не найдена колонка «оплачено»"):
            pc.read_osv_cells(path)


# ---------------------------------------------------------------------------
# Панель = исходные файлы
# ---------------------------------------------------------------------------

class TestCompare:
    def test_panel_matches_files(self, built):
        table, summary, _, bad = pc.run_check(*built)
        inside = table[table["in_panel"]]
        assert inside["osv_ok"].all() and inside["payments_ok"].all()
        assert bad[~bad["osv_ok"] | ~bad["payments_ok"]].empty
        assert summary.set_index("Проверка").loc["Сальдо, начислено, оплачено, dz: ОСВ = панель", "Расхождений"] == 0

    @pytest.mark.parametrize("column, value", [("saldo_start", 999.0), ("accrual", 1.0), ("paid", 7.0), ("dz", 1.0)])
    def test_wrong_panel_value_is_found(self, built, column, value):
        p, data_dir = built
        p = p.copy()
        row = p.index[(p["ls"] == 20) & (p["month"] == T("2025-02-01"))][0]
        p.loc[row, column] = value
        _, _, _, bad = pc.run_check(p, data_dir)
        assert bad.loc[~bad["osv_ok"], ["ls", "month"]].values.tolist() == [[20, T("2025-02-01")]]

    def test_wrong_payments_are_found(self, built):
        p, data_dir = built
        p = p.copy()
        p.loc[(p["ls"] == 10) & (p["month"] == T("2025-01-01")), "paid_payments"] = 0.0
        _, _, _, bad = pc.run_check(p, data_dir)
        assert bad.loc[~bad["payments_ok"], "ls"].tolist() == [10]

    def test_lost_account_is_reported(self, built):
        p, data_dir = built
        _, summary, _, _ = pc.run_check(p[p["ls"] != 20], data_dir)
        row = summary.set_index("Проверка").loc["ЛС из ОСВ есть в панели"]
        # ЛС 20 с долгом пропал — это не «без движения», объяснения нет.
        assert row["Расхождений"] == 1 and row["Из них объяснено"] == 0


# ---------------------------------------------------------------------------
# Причины расхождений «Оплачено» ОСВ ↔ оплаты 03
# ---------------------------------------------------------------------------

def reasons(rows, december=0.0):
    """rows — по месяцам с 2025-01: (сальдо, начислено, «Оплачено» ОСВ, оплаты 03). Возвращает причины по месяцам."""
    months = [T("2025-01-01") + pd.DateOffset(months=i) for i in range(len(rows))]
    cells = pd.DataFrame({
        "ls": 1, "row": 3, "month": months,
        "saldo_start": [r[0] for r in rows], "accrual": [r[1] for r in rows], "paid": [r[2] for r in rows],
        "saldo_start_cell": "B3", "accrual_cell": "C3", "paid_cell": "D3",
    })
    payments = pd.DataFrame(
        [(1, m + pd.Timedelta(days=10), r[3]) for m, r in zip(months, rows) if r[3]]
        + ([(1, T("2024-12-30"), december)] if december else []),
        columns=["ls", "payment_date", "amount"],
    )
    payments["month"] = payments["payment_date"].dt.to_period("M").dt.to_timestamp()
    p = cells[["ls", "month", "saldo_start", "accrual", "paid"]].assign(
        dz=lambda d: d["saldo_start"].clip(lower=0), paid_payments=[r[3] for r in rows])
    table = pc.compare(p, cells, payments)
    return [None if pd.isna(r) else r for r in table["paid_reason"]]


class TestReasons:
    def test_equal_sums(self):
        assert reasons([(0, 100, 100, 100), (0, 100, 100, 100)]) == [None, None]

    def test_shift_to_next_month(self):
        # Оплата 100 от конца февраля в ОСВ проведена мартом.
        rows = [(0, 100, 100, 100), (0, 100, 0, 100), (100, 100, 200, 100), (0, 100, 100, 100)]
        assert reasons(rows) == [None, pc.SHIFT, pc.SHIFT, None]

    def test_shift_on_top_of_old_difference(self):
        # С января разница −100 (оплата декабря 2024), потом 10 ₽ апреля проведены в мае.
        rows = [(100, 100, 100, 0), (100, 100, 100, 100), (100, 100, 100, 110), (90, 100, 100, 90), (100, 100, 100, 100)]
        assert reasons(rows, december=100) == [pc.BEFORE_DATA, None, pc.SHIFT, pc.SHIFT, None]

    def test_rolling_shift_up_to_three_months(self):
        # Февральская и мартовская оплаты проведены следующими месяцами: разница сходится в апреле.
        rows = [(0, 100, 100, 100), (0, 100, 0, 100), (100, 100, 100, 100), (100, 100, 200, 100)]
        assert reasons(rows) == [None, pc.SHIFT, None, pc.SHIFT]
        rows = [(0, 100, 100, 100), (0, 100, 0, 100), (100, 100, 50, 100), (150, 100, 250, 100)]
        assert reasons(rows) == [None, pc.SHIFT, pc.SHIFT, pc.SHIFT]

    def test_longer_chain_is_not_a_shift(self):
        # Разница сходится только через 4 месяца — длиннее окна переноса.
        rows = [(0, 100, 0, 100), (100, 100, 50, 100), (150, 100, 50, 100), (200, 100, 300, 100)]
        assert pc.UNEXPLAINED in reasons(rows)

    def test_december_payment_in_january(self):
        rows = [(100, 100, 100, 0), (100, 100, 100, 100)]
        assert reasons(rows, december=100) == [pc.BEFORE_DATA, None]
        assert reasons(rows) == [pc.UNEXPLAINED, None]  # без оплаты декабря — не объяснено

    def test_saldo_confirms_payments_file(self):
        # «Оплачено» ОСВ 300, по файлу 03 — 100. Сальдо марта 100 = 100 + 100 − 100: верен файл 03.
        rows = [(0, 100, 100, 100), (100, 100, 300, 100), (100, 100, 100, 100)]
        assert reasons(rows) == [None, pc.CONFIRMED_BY_SALDO, None]

    def test_end_of_data(self):
        rows = [(0, 100, 100, 100), (0, 100, 0, 100)]
        assert reasons(rows) == [None, pc.END_OF_DATA]


# ---------------------------------------------------------------------------
# 20 ЛС и команда
# ---------------------------------------------------------------------------

class TestCommand:
    @pytest.fixture
    def folder(self, built, tmp_path):
        p, _ = built
        target = tmp_path / "out" / "2025-03"
        target.mkdir(parents=True)
        panel.write_parquet(p, target / "panel.parquet")
        return target

    def test_writes_workbook(self, built, folder, tmp_path):
        _, data_dir = built
        code = pc.main(["--month", "2025-03", "--data", str(data_dir), "--out", str(tmp_path / "out")])
        assert code in (0, 2)  # 2 — есть необъяснённые расхождения «Оплачено» ↔ 03, они в отдельном листе
        book = pd.read_excel(folder / "panel_check.xlsx", sheet_name=None)
        assert list(book) == ["Как проверить", "Итог", "20 ЛС", "Не объяснено"]
        manual = book["20 ЛС"]
        assert set(manual["ЛС"]) == {10, 20, 30}
        assert (manual["02 = панель"] == "да").all() and (manual["03 = панель"] == "да").all()
        assert manual.loc[(manual["ЛС"] == 20) & (manual["Месяц"] == "2025-02"), "Ячейка «СЗ на начало»"].tolist() == ["E4"]

    def test_chosen_accounts_and_errors(self, built, folder, tmp_path, caplog):
        _, data_dir = built
        args = ["--month", "2025-03", "--data", str(data_dir), "--out", str(tmp_path / "out")]
        pc.main(args + ["--ls", "30"])
        assert set(pd.read_excel(folder / "panel_check.xlsx", sheet_name="20 ЛС")["ЛС"]) == {30}
        with caplog.at_level(logging.ERROR):
            assert pc.main(args + ["--ls", "999"]) == 1
            assert pc.main(["--month", "2025-04", "--data", str(data_dir), "--out", str(tmp_path / "out")]) == 1
        assert "В ОСВ нет ЛС: 999" in caplog.text and "Сначала постройте панель" in caplog.text


@pytest.mark.real_data
def test_real_data_reconciliation(real_data_dir, settings):
    raw, _ = quality.clean(loaders.load_all(real_data_dir, settings, loaders.default_cache_dir(settings)), settings)
    p = panel.build_from_raw(raw, settings)["panel"]
    table, summary, manual, bad = pc.run_check(p, real_data_dir)
    inside = table[table["in_panel"]]
    # Панель совпадает с каждой ячейкой ОСВ и с суммами оплат файла 03.
    assert inside["osv_ok"].all() and inside["payments_ok"].all()
    # ЛС, которых нет в панели, — все без движения (убраны очисткой).
    row = summary.set_index("Проверка").loc["ЛС из ОСВ есть в панели"]
    assert row["Расхождений"] == row["Из них объяснено"] > 0
    # «Оплачено» ОСВ ↔ 03: необъяснённых строк меньше 0,01 % и меньше 5 % всех расхождений.
    reasons = inside["paid_reason"].value_counts()
    assert reasons.get(pc.UNEXPLAINED, 0) < 0.0001 * len(inside)
    assert reasons.get(pc.UNEXPLAINED, 0) < 0.05 * reasons.sum()
    assert manual["ЛС"].nunique() == 20 and len(bad) == reasons.get(pc.UNEXPLAINED, 0)
