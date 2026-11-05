"""Признаки для правил (DATA-7): норматив P-07 (выгрузка заказчика или прокси) и месяцы без движения.

Главная проверка — условия R-VST-05, R-NTC-01 и T-1→2 из таблицы правил v0.3 вычисляются
на колонках панели: параметры P-08 и P-12 берутся из таблицы, поля — из panel.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from tns.data import loaders, panel, quality
from tns.data.validator import ValidationError
from tns.rules.loader import _condition_identifiers, load_rulebook
from tests.conftest import make_events, make_info, make_osv, make_payments

RULES_V03 = loaders.REPO_ROOT / "docs" / "Правила воздействия ТНС v0.3.xlsx"
# Поля условий, которые считают другие части системы (стадии, журнал мер), а не панель.
FROM_OTHER_TABLES = {"notice_date", "months_in_stage"}
T = pd.Timestamp


@pytest.fixture(scope="module")
def book():
    if not RULES_V03.exists():
        pytest.skip("Нет таблицы правил v0.3.")
    return load_rulebook(RULES_V03)


def conditions(book) -> dict[str, str]:
    transitions = {t.code: t.condition for t in book.transitions}
    return {"R-VST-05": book.rules["R-VST-05"].condition, "R-NTC-01": book.rules["R-NTC-01"].condition,
            "T-1→2": transitions["T-1→2"]}


def panel_clauses(condition: str, columns, params) -> list[str]:
    """Части условия (через AND), которые считаются только по колонкам панели; P-xx заменены значениями."""
    result = []
    for clause in re.split(r"\s+AND\s+", condition):
        if re.search(r"\bNOT\b|IS NULL|result\(|EX-\d+", clause):
            continue
        names = _condition_identifiers(clause)
        if names and names <= set(columns):
            result.append(re.sub(r"P-\d+", lambda m: repr(params[m.group(0)].value_num), clause))
    return result


def evaluate(table: pd.DataFrame, clauses: list[str]) -> pd.Series:
    """Все части условия на строках панели. Пустой норматив или признак — условие не выполнено."""
    result = pd.Series(True, index=table.index)
    numeric = table.astype({c: "float64" for c in ["dz", "norm", "months_no_payment"]})
    for clause in clauses:
        result &= numeric.eval(clause).fillna(False).astype(bool)
    return result


def build(osv, payments=None, norm=None, settings=None):
    settings = settings or loaders.load_settings()
    raw = {
        "info": make_info(sorted(osv["ls"].unique())), "osv": osv,
        "payments": payments if payments is not None else make_payments([]),
        "events": make_events([]), "norm": norm,
    }
    return panel.build_from_raw(raw, settings)["panel"]


def at(table, ls, month, name):
    value = table.set_index(["ls", "month"]).loc[(ls, T(month)), name]
    return None if pd.isna(value) else value


# ---------------------------------------------------------------------------
# Поля условий есть в панели
# ---------------------------------------------------------------------------

class TestRuleFields:
    def test_every_field_is_in_panel_or_documented_as_missing(self, book):
        available = {f_name: f.available for f in book.fields for f_name in f.names}
        for code, condition in conditions(book).items():
            for name in _condition_identifiers(condition):
                if name in panel.PANEL_COLUMNS or name in FROM_OTHER_TABLES:
                    continue
                # Остальные поля в данных отсутствуют, и это записано в листе «Поля данных».
                assert str(available.get(name, "")).startswith("Нет"), f"{code}: поля «{name}» нет в панели"

    def test_panel_parts_of_conditions(self, book):
        clauses = {code: panel_clauses(c, panel.PANEL_COLUMNS, book.params) for code, c in conditions(book).items()}
        assert clauses["R-VST-05"] == ["months_no_payment > 12.0"]
        assert clauses["R-NTC-01"] == ["dz > 2.0 * norm"]
        assert "dz > 2.0 * norm" in clauses["T-1→2"] and "dz >= 1500" in clauses["T-1→2"]


# ---------------------------------------------------------------------------
# R-NTC-01: dz > P-08 × norm, прокси норматива
# ---------------------------------------------------------------------------

class TestNoticeCondition:
    @pytest.fixture
    def table(self):
        # Начисления 100, 200, 300 → на 01.04 прокси = 200. Двукратный — 400.
        return build(make_osv({
            1: [(0, 100, 0), (100, 200, 0), (300, 300, 0), (600, 0, 0)],          # долг 600 > 400
            2: [(0, 100, 100), (0, 200, 200), (0, 300, 0), (300, 0, 0)],          # долг 300 < 400
            3: [(0, 100, 0), (100, 300, 0), (400, 0, 0), (400, 0, 0)],            # долг 400 = 400, не больше
            4: [(0, 100, 0), (100, -500, 0), (0, 100, 0), (100, 0, 0)],           # среднее −100 → не определить
        }))

    def test_proxy_values(self, table):
        assert at(table, 1, "2025-04-01", "norm") == 200 and at(table, 1, "2025-04-01", "norm_source") == "proxy"
        assert at(table, 3, "2025-04-01", "norm") == pytest.approx(400 / 3)
        # Меньше трёх месяцев истории — прокси нет.
        assert at(table, 1, "2025-03-01", "norm") is None and at(table, 1, "2025-03-01", "norm_source") is None

    def test_non_positive_proxy_is_unknown(self, table):
        assert at(table, 4, "2025-04-01", "norm") is None
        assert at(table, 4, "2025-04-01", "norm_source") is None

    def test_condition(self, table, book):
        clauses = panel_clauses(book.rules["R-NTC-01"].condition, panel.PANEL_COLUMNS, book.params)
        april = table[table["month"] == T("2025-04-01")].set_index("ls")
        assert evaluate(april, clauses).to_dict() == {1: True, 2: False, 3: True, 4: False}
        # ЛС 3: 400 > 2 × 133,33 — да. ЛС 4: норматив не определён — уведомление не назначается.


# ---------------------------------------------------------------------------
# R-VST-05: months_no_payment > P-12
# ---------------------------------------------------------------------------

def test_visit_condition_after_12_months_without_payments(book):
    osv = make_osv({1: [(100, 0, 0)] * 15, 2: [(100, 0, 0)] * 15})
    payments = make_payments([(2, "2025-06-10", 50, 5)])
    table = build(osv, payments)
    clauses = panel_clauses(book.rules["R-VST-05"].condition, panel.PANEL_COLUMNS, book.params)
    hit = table.assign(hit=evaluate(table, clauses)).pivot(index="month", columns="ls", values="hit")
    # ЛС 1 без оплат с начала данных: на 01.02.2026 — 13 мес. (> 12), на 01.01.2026 — 12 (не больше 12).
    assert at(table, 1, "2026-01-01", "months_no_payment") == 12 and not hit.loc[T("2026-01-01"), 1]
    assert at(table, 1, "2026-02-01", "months_no_payment") == 13 and hit.loc[T("2026-02-01"), 1]
    # ЛС 2 платил в июне 2025: к марту 2026 без оплат 8 мес.
    assert at(table, 2, "2026-03-01", "months_no_payment") == 8 and not hit[2].any()


# ---------------------------------------------------------------------------
# Выгрузка норматива заказчика
# ---------------------------------------------------------------------------

class TestCustomerNorm:
    def write(self, folder, rows, name="Норматив.xlsx", columns=("ЛС", "Месяц", "Норматив")):
        frame = pd.DataFrame(rows, columns=list(columns))
        path = folder / name
        if path.suffix == ".csv":
            frame.to_csv(path, sep=";", index=False, decimal=",")
        else:
            frame.to_excel(path, index=False)
        return path

    def read(self, folder, settings, **options):
        settings["norm"].update(customer_file=options.pop("file", "Норматив.xlsx"), **options)
        return loaders.read_customer_norm(folder, settings)

    def test_not_configured_gives_empty_table(self, data_dir, settings):
        norm = loaders.read_customer_norm(data_dir, settings)
        assert norm.empty
        loaders.check_schema("norm", norm)

    def test_month_column_is_optional(self, tmp_path, settings):
        self.write(tmp_path, [(1, 500), (2, 700.5)], columns=("ЛС", "Норматив"))
        norm = self.read(tmp_path, settings)
        loaders.check_schema("norm", norm)
        assert norm["norm"].tolist() == [500, 700.5] and norm["month"].isna().all()

    def test_csv_with_comma_and_multiplier(self, tmp_path, settings):
        self.write(tmp_path, [(1, "2025-03-01", 1000.5)], name="норматив.csv")
        norm = self.read(tmp_path, settings, file="норматив.csv", customer_multiplier=2)
        assert norm["norm"].tolist() == [500.25] and norm["month"].tolist() == [T("2025-03-01")]

    def test_column_names_from_settings(self, tmp_path, settings):
        self.write(tmp_path, [(1, 300)], columns=("Лицевой счёт", "Норматив, руб."))
        norm = self.read(tmp_path, settings, customer_columns={"ls": "Лицевой счёт", "norm": "Норматив, руб."})
        assert norm["norm"].tolist() == [300]

    def test_missing_column(self, tmp_path, settings):
        self.write(tmp_path, [(1, 300)], columns=("ЛС", "Сумма"))
        with pytest.raises(ValueError, match="нет колонки «Норматив».*norm.customer_columns"):
            self.read(tmp_path, settings)

    def test_repeated_month(self, tmp_path, settings):
        self.write(tmp_path, [(1, "2025-03-01", 300), (1, "2025-03-15", 310)])
        with pytest.raises(ValueError, match="норматив указан несколько раз на один месяц, например ЛС 1 за 2025-03"):
            self.read(tmp_path, settings)

    def test_bad_values_are_dropped_with_warning(self, tmp_path, settings):
        rows = [(ls, "2025-03-01", 300) for ls in range(1, 10)] + [(10, "2025-03-01", "нет"), (11, "2025-03-01", -5)]
        self.write(tmp_path, rows)
        norm = self.read(tmp_path, settings)
        assert len(norm) == 9
        assert "положительное число" in norm.attrs["problems"][0].message

    def test_missing_file_stops_loading(self, data_dir, settings):
        settings["norm"]["customer_file"] = "нет такого.xlsx"
        with pytest.raises(ValidationError, match="нет такого.xlsx"):
            loaders.load_all(data_dir, settings)

    def test_bad_multiplier(self, tmp_path, settings):
        self.write(tmp_path, [(1, 300)], columns=("ЛС", "Норматив"))
        with pytest.raises(ValueError, match="norm.customer_multiplier"):
            self.read(tmp_path, settings, customer_multiplier=0)

    def test_panel_takes_customer_value_where_it_exists(self, book):
        osv = make_osv({ls: [(0, 100, 0), (100, 100, 0), (200, 100, 0), (300, 100, 0), (400, 100, 0)] for ls in (1, 2, 3)})
        customer = pd.DataFrame({
            "ls": [1, 2, 2], "month": [pd.NaT, T("2025-02-01"), T("2025-04-01")], "norm": [150.0, 120.0, 250.0],
        }).astype({"month": "datetime64[ns]"})
        table = build(osv, norm=customer)
        # ЛС 1: значение без месяца — для всех месяцев.
        assert [at(table, 1, m, "norm") for m in ["2025-01-01", "2025-05-01"]] == [150, 150]
        # ЛС 2: с февраля 120, с апреля 250; в январе выгрузки нет и прокси тоже (мало истории).
        assert [at(table, 2, f"2025-0{m}-01", "norm") for m in range(1, 6)] == [None, 120, 120, 250, 250]
        assert [at(table, 2, f"2025-0{m}-01", "norm_source") for m in range(1, 6)] == [None] + ["customer"] * 4
        # ЛС 3: выгрузки нет — прокси.
        assert at(table, 3, "2025-05-01", "norm") == 100 and at(table, 3, "2025-05-01", "norm_source") == "proxy"
        # Условие R-NTC-01 работает и с нормативом заказчика: на 01.05 долг 400; 400 > 2×150, 400 < 2×250.
        clauses = panel_clauses(book.rules["R-NTC-01"].condition, panel.PANEL_COLUMNS, book.params)
        may = table[table["month"] == T("2025-05-01")].set_index("ls")
        assert evaluate(may, clauses).to_dict() == {1: True, 2: False, 3: True}

    def test_load_all_and_panel_with_customer_file(self, data_dir, settings):
        info = loaders.read_info(data_dir)
        self.write(data_dir, [(int(info["ls"].iloc[0]), 321)], columns=("ЛС", "Норматив"))
        settings["norm"]["customer_file"] = "Норматив.xlsx"
        raw, _ = quality.clean(loaders.load_all(data_dir, settings), settings)
        table = panel.build_from_raw(raw, settings)["panel"]
        first = table[table["ls"] == info["ls"].iloc[0]]
        assert (first["norm"] == 321).all() and (first["norm_source"] == "customer").all()


@pytest.mark.real_data
def test_real_data_rule_features(real_data_dir, settings, book):
    raw, _ = quality.clean(loaders.load_all(real_data_dir, settings, loaders.default_cache_dir(settings)), settings)
    table = panel.build_from_raw(raw, settings)["panel"]
    assert raw["norm"].empty and set(table["norm_source"].dropna()) == {"proxy"}
    assert (table["norm"].dropna() > 0).all()
    # Прокси есть с 4-го месяца данных (минимум 3 месяца истории) почти у всех ЛС.
    later = table[table["month"] >= T("2025-04-01")]
    assert later["norm"].notna().mean() > 0.9
    for code, condition in conditions(book).items():
        clauses = panel_clauses(condition, panel.PANEL_COLUMNS, book.params)
        assert clauses, code
        hits = evaluate(table, clauses)
        assert 0 < hits.mean() < 1, code
