"""Тесты панели «ЛС × месяц», анкеты и журнала мер (DATA-4, tns/data/panel.py).

Ожидаемые значения посчитаны вручную по docs/interfaces.md, разделы 1–3.
Настройки — из configs/settings.yaml: доля 0.30, окно до конца месяца, FIFO, норматив по 12 мес. (минимум 3).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import yaml

from tns import pipeline
from tns.data import loaders, panel, quality
from tests.conftest import LIMITS_YAML, make_events, make_info, make_osv, make_payments

M = [pd.Timestamp("2025-01-01") + pd.DateOffset(months=i) for i in range(6)]


@pytest.fixture
def settings():
    return loaders.load_settings()


def build(osv, payments=None, events=None, info=None, settings=None):
    settings = settings or loaders.load_settings()
    raw = {
        "info": info if info is not None else make_info(sorted(osv["ls"].unique())),
        "osv": osv,
        "payments": payments if payments is not None else make_payments([]),
        "events": events if events is not None else make_events([]),
    }
    return panel.build_from_raw(raw, settings)


def column(tables, ls, name):
    rows = tables["panel"][tables["panel"]["ls"] == ls].sort_values("month")
    return [None if pd.isna(v) else v for v in rows[name]]


# ---------------------------------------------------------------------------
# Формат по контракту
# ---------------------------------------------------------------------------

class TestContract:
    @pytest.fixture
    def tables(self):
        osv = make_osv({1: [(0, 100, 0), (100, 100, 0)], 2: [(50, 10, 10), (50, 10, 10)]})
        events = make_events([(1, "sms", "2025-02-01"), (2, "restriction", "2025-01-20")])
        return build(osv, make_payments([(2, "2025-01-05", 10, 5)]), events)

    @pytest.mark.parametrize("name", ["ls_info", "panel", "measure_events"])
    def test_columns_order_and_types(self, tables, name):
        expected = panel.OUTPUT_SCHEMAS[name]
        assert list(tables[name].columns) == list(expected)
        for column_name, dtype in expected.items():
            actual = tables[name][column_name].dtype
            if isinstance(dtype, list):
                assert list(actual.categories) == dtype, column_name
            else:
                assert str(actual) == dtype, column_name

    def test_one_row_per_ls_and_month(self, tables):
        p = tables["panel"]
        assert not p.duplicated(["ls", "month"]).any()
        assert (p["month"].dt.day == 1).all()
        assert p.groupby("ls").size().tolist() == [2, 2]

    def test_contract_types_survive_parquet(self, tables, tmp_path):
        for name, table in tables.items():
            panel.write_parquet(table, tmp_path / f"{name}.parquet")
            pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / f"{name}.parquet"), table)

    def test_wrong_order_is_caught(self, tables):
        shuffled = tables["panel"][list(reversed(panel.PANEL_COLUMNS))]
        with pytest.raises(ValueError, match="порядок колонок не как в контракте"):
            loaders.check_schema("panel", shuffled, panel.OUTPUT_SCHEMAS, panel.OUTPUT_KEYS["panel"])

    def test_duplicate_keys_are_caught(self, tables):
        doubled = pd.concat([tables["panel"], tables["panel"].head(1)], ignore_index=True)
        with pytest.raises(ValueError, match=r"повторяются ключи \(ls, month\): 1 строк"):
            loaders.check_schema("panel", doubled, panel.OUTPUT_SCHEMAS, panel.OUTPUT_KEYS["panel"])

    def test_start_columns_list(self):
        assert set(panel.PANEL_START_COLUMNS) | {"ls", "month", "accrual", "paid", "paid_payments", "pay30_m"} == set(panel.PANEL_COLUMNS)


# ---------------------------------------------------------------------------
# ДЗ, начислено, оплачено
# ---------------------------------------------------------------------------

class TestDebtAndMoney:
    def test_dz_is_positive_part_of_saldo(self):
        tables = build(make_osv({1: [(-20, 10, 0), (0, 10, 0), (35.5, 10, 0)]}))
        assert column(tables, 1, "saldo_start") == [-20, 0, 35.5]
        assert column(tables, 1, "dz") == [0, 0, 35.5]

    def test_accrual_and_paid_from_osv(self):
        tables = build(make_osv({1: [(0, 120.5, 0), (120.5, 130, 100)]}))
        assert column(tables, 1, "accrual") == [120.5, 130]
        assert column(tables, 1, "paid") == [0, 100]

    def test_paid_payments_from_file_03(self):
        payments = make_payments([
            (1, "2025-01-05", 40, 5), (1, "2025-01-25", 20, 5),  # январь: 60
            (1, "2025-02-10", 50, 5), (1, "2025-02-11", -50, 5),  # февраль: сторно гасит оплату
            (1, "2024-12-30", 999, 5), (1, "2025-04-01", 999, 5),  # вне периода ОСВ
            (7, "2025-01-05", 999, 5),  # ЛС нет в ОСВ
        ])
        tables = build(make_osv({1: [(100, 0, 0), (100, 0, 0), (100, 0, 0)]}), payments)
        assert column(tables, 1, "paid_payments") == [60, 0, 0]

    def test_pay30_threshold(self):
        osv = make_osv({1: [(100, 0, 0)], 2: [(100, 0, 0)], 3: [(33.33, 0, 0)], 4: [(33.33, 0, 0)], 5: [(0, 0, 0)], 6: [(-5, 0, 0)]})
        payments = make_payments([
            (1, "2025-01-31", 30, 5), (2, "2025-01-31", 29.99, 5),  # 30 % от 100 = 30,00
            (3, "2025-01-31", 10, 5), (4, "2025-01-31", 9.99, 5),  # 30 % от 33,33 = 9,999 → 10,00
            (5, "2025-01-31", 10, 5), (6, "2025-01-31", 10, 5),  # долга нет — <NA>
        ])
        values = build(osv, payments)["panel"].set_index("ls")["pay30_m"]
        assert [values[i] for i in (1, 2, 3, 4)] == [True, False, True, False]
        assert values.loc[[5, 6]].isna().all()

    def test_pay30_share_from_settings(self, settings):
        settings["result"]["share"] = 0.5
        values = build(make_osv({1: [(100, 0, 0)]}), make_payments([(1, "2025-01-10", 30, 5)]), settings=settings)
        assert column(values, 1, "pay30_m") == [False]


# ---------------------------------------------------------------------------
# Просрочка (P-02, FIFO), месяцы без оплат, норматив
# ---------------------------------------------------------------------------

class TestOverdue:
    def test_new_debt_grows_month_by_month(self):
        # Начисляем 100 каждый месяц, не платим. На 01.M счёт за M−1 ещё не просрочен,
        # остальной долг покрывается счетами M−2, M−3, … — по месяцу за каждый.
        osv = make_osv({1: [(0, 100, 0), (100, 100, 0), (200, 100, 0), (300, 100, 0), (400, 100, 0)]})
        tables = build(osv)
        assert column(tables, 1, "dz_overdue") == [0, 0, 100, 200, 300]
        assert column(tables, 1, "overdue_m") == [0, 0, 1, 2, 3]
        assert column(tables, 1, "overdue_censored") == [False] * 5

    def test_debt_older_than_data_is_censored(self):
        osv = make_osv({1: [(500, 100, 100), (500, 100, 100), (500, 100, 100)]})
        tables = build(osv)
        # Январь: начисления за декабрь нет — срок не определить.
        assert column(tables, 1, "dz_overdue") == [None, 400, 400]
        assert column(tables, 1, "overdue_m") == [None, 1, 2]
        assert column(tables, 1, "overdue_censored") == [True, True, True]

    def test_paid_off_debt_has_no_overdue(self):
        osv = make_osv({1: [(0, 100, 0), (100, 100, 200), (0, 100, 0)]})
        assert column(build(osv), 1, "overdue_m") == [0, 0, 0]

    def test_zero_accrual_months_count_or_skip(self, settings):
        # Март: долг 200, счёт февраля 100 не просрочен; остаток 100 покрывает январь через пустой месяц.
        osv = make_osv({1: [(0, 100, 0), (100, 0, 0), (100, 100, 0), (200, 0, 0)]})
        assert column(build(osv, settings=settings), 1, "overdue_m")[3] == 2
        settings["overdue"]["zero_accrual_months"] = "skip"
        assert column(build(osv, settings=settings), 1, "overdue_m")[3] == 1

    def test_negative_accrual_does_not_cover_debt(self):
        osv = make_osv({1: [(0, 100, 0), (100, -30, 0), (70, 0, 0)]})
        assert column(build(osv), 1, "dz_overdue") == [0, 0, 70]


class TestMonthsWithoutPayment:
    def test_run_and_censoring(self):
        osv = make_osv({1: [(100, 0, 0)] * 5})
        payments = make_payments([(1, "2025-02-10", 50, 5)])
        tables = build(osv, payments)
        # Оплата в феврале: в марте серия обнуляется, до февраля тянется с начала данных.
        assert column(tables, 1, "months_no_payment") == [0, 1, 0, 1, 2]
        assert column(tables, 1, "months_no_payment_censored") == [True, True, False, False, False]

    def test_reversal_only_month_is_no_payment(self):
        payments = make_payments([(1, "2025-01-10", 50, 5), (1, "2025-02-10", 50, 5), (1, "2025-02-11", -50, 5)])
        tables = build(make_osv({1: [(100, 0, 0)] * 3}), payments)
        assert column(tables, 1, "months_no_payment") == [0, 0, 1]


class TestNorm:
    def test_proxy_needs_three_months_before(self):
        osv = make_osv({1: [(0, a, 0) for a in (100, 200, 300, 400, 500)]})
        tables = build(osv)
        assert column(tables, 1, "norm") == [None, None, None, 200, 250]
        assert column(tables, 1, "norm_source") == [None, None, None, "proxy", "proxy"]

    def test_proxy_window_from_settings(self, settings):
        settings["norm"]["proxy_months"] = 3
        osv = make_osv({1: [(0, a, 0) for a in (100, 200, 300, 400, 500)]})
        assert column(build(osv, settings=settings), 1, "norm")[4] == 300


# ---------------------------------------------------------------------------
# Правило known_at: колонки start месяца M не зависят от данных за M и позже
# ---------------------------------------------------------------------------

def test_start_columns_do_not_see_the_future():
    rng = np.random.default_rng(7)
    rows = {ls: [(float(rng.integers(-50, 500)), float(rng.integers(0, 200)), float(rng.integers(0, 200)))
                 for _ in range(6)] for ls in range(1, 41)}
    pay_rows = [(ls, f"2025-{m:02d}-{d:02d}", float(rng.integers(-20, 300)), 5)
                for ls in range(1, 41) for m in range(1, 7) for d in (5, 20) if rng.random() < 0.6]
    osv, payments = make_osv(rows), make_payments(pay_rows)
    cut = M[3]  # срез 01.04.2025
    before = build(osv, payments)["panel"]

    future = osv.copy()
    future.loc[future["month"] >= cut, ["accrual", "paid"]] *= 3  # начисления и оплаты за M и позже
    future.loc[future["month"] > cut, "saldo_start"] += 1000  # сальдо после 01.M
    changed_payments = payments.copy()
    changed_payments.loc[changed_payments["payment_date"] >= cut, "amount"] += 777
    after = build(future, changed_payments)["panel"]

    known = before["month"] <= cut
    pd.testing.assert_frame_equal(
        before.loc[known, ["ls", "month", *panel.PANEL_START_COLUMNS]].reset_index(drop=True),
        after.loc[known, ["ls", "month", *panel.PANEL_START_COLUMNS]].reset_index(drop=True),
    )
    # А колонки end за M меняются — значит, данные за M правда другие.
    at_cut = before["month"] == cut
    assert not before.loc[at_cut, "paid_payments"].equals(after.loc[at_cut, "paid_payments"])


# ---------------------------------------------------------------------------
# Анкета
# ---------------------------------------------------------------------------

class TestLsInfo:
    def test_ls_only_in_osv_get_unknown_flags(self):
        info = make_info([1])
        info["has_phone"] = pd.array([True], dtype="boolean")
        tables = build(make_osv({1: [(0, 0, 0)], 2: [(10, 0, 0)]}), info=info)
        result = tables["ls_info"].set_index("ls")
        assert result.loc[1, "has_phone"] == True  # noqa: E712
        assert result.loc[2, ["has_phone", "house_type", "address_guid"]].isna().all()
        assert list(tables["ls_info"]["ls"]) == [1, 2]


# ---------------------------------------------------------------------------
# Журнал мер: какие меры применялись в месяце и их результат
# ---------------------------------------------------------------------------

class TestMeasureEvents:
    @pytest.fixture
    def osv(self):
        return make_osv({1: [(100, 0, 0), (100, 0, 0), (100, 0, 0)], 2: [(0, 0, 0), (0, 0, 0), (0, 0, 0)]})

    def test_one_row_per_measure_and_month(self, osv):
        events = make_events([(1, "sms", "2025-02-01"), (1, "sms", "2025-02-01"), (1, "visit", "2025-02-20"),
                              (1, "visit", "2025-02-03"), (1, "sms", "2025-03-01")])
        log = build(osv, events=events)["measure_events"]
        assert log[["measure", "month", "n_raw"]].astype({"measure": str}).values.tolist() == [
            ["sms", M[1], 2], ["sms", M[2], 1], ["visit", M[1], 2],
        ]
        assert log.loc[log["measure"] == "visit", "event_date"].tolist() == [pd.Timestamp("2025-02-03")]
        assert log["date_precision"].astype(str).tolist() == ["month", "month", "month"]

    def test_measures_of_a_month_join_the_panel(self, osv):
        events = make_events([(1, "sms", "2025-02-01"), (1, "claim", "2025-02-01")])
        tables = build(osv, events=events)
        joined = tables["panel"].merge(tables["measure_events"], on=["ls", "month"], how="left")
        assert sorted(joined.loc[(joined["ls"] == 1) & (joined["month"] == M[1]), "measure"].astype(str)) == ["claim", "sms"]

    def test_result_window_for_day_and_month_precision(self, osv):
        payments = make_payments([(1, "2025-02-05", 30, 5), (1, "2025-02-28", 10, 5), (1, "2025-03-01", 500, 5)])
        events = make_events([(1, "sms", "2025-02-01"), (1, "restriction", "2025-02-20")])
        log = build(osv, payments, events)["measure_events"].set_index("measure")
        # СМС: точность «месяц», окно с 01.02 по 28.02 включительно: 30 + 10.
        assert log.loc["sms", "paid_in_window"] == 40 and log.loc["sms", "result_30pct"] == True  # noqa: E712
        # Ограничение 20.02: оплата 05.02 была раньше меры, в окно входит только 28.02.
        assert log.loc["restriction", "paid_in_window"] == 10 and log.loc["restriction", "result_30pct"] == False  # noqa: E712
        assert log.loc["restriction", "window_end"] == pd.Timestamp("2025-02-28")
        assert log.loc["restriction", "dz_at_event"] == 100

    def test_month_precision_window_starts_on_the_1st(self, osv):
        # Выезд — мера с точностью «месяц», даже если в файле стоит 15-е: окно открывается с 01.02.
        payments = make_payments([(1, "2025-02-05", 30, 5)])
        log = build(osv, payments, make_events([(1, "visit", "2025-02-15")]))["measure_events"]
        assert log["event_date"].tolist() == [pd.Timestamp("2025-02-15")]
        assert log["paid_in_window"].tolist() == [30] and log["result_30pct"].tolist() == [True]

    def test_next_month_window(self, osv, settings):
        settings["result"]["window"] = "next_month_end"
        payments = make_payments([(1, "2025-03-15", 30, 5)])
        log = build(osv, payments, make_events([(1, "claim", "2025-02-01")]), settings=settings)["measure_events"]
        assert log["window_end"].tolist() == [pd.Timestamp("2025-03-31")]
        assert log["paid_in_window"].tolist() == [30] and log["result_30pct"].tolist() == [True]

    def test_no_debt_gives_unknown_result(self, osv):
        log = build(osv, make_payments([(2, "2025-02-10", 5, 5)]), make_events([(2, "sms", "2025-02-01")]))["measure_events"]
        assert log["dz_at_event"].tolist() == [0] and log["result_30pct"].isna().all()

    def test_event_outside_panel(self, osv, caplog):
        with caplog.at_level(logging.WARNING):
            log = build(osv, events=make_events([(1, "sms", "2025-06-01")]))["measure_events"]
        assert log["dz_at_event"].isna().all() and log["result_30pct"].isna().all()
        assert "без долга на месяц меры" in caplog.text


# ---------------------------------------------------------------------------
# Одна команда
# ---------------------------------------------------------------------------

class TestOneCommand:
    @pytest.fixture
    def config(self, settings, tmp_path):
        fallback = tmp_path / "limits.yaml"
        fallback.write_text(yaml.safe_dump(LIMITS_YAML), encoding="utf-8")
        settings["paths"] = {**settings["paths"], "cache_dir": str(tmp_path / "cache"), "limits_fallback": str(fallback)}
        path = tmp_path / "settings.yaml"
        path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")
        return path

    def test_panel_command(self, data_dir, config, tmp_path):
        out = tmp_path / "out"
        assert panel.main(["--data", str(data_dir), "--out", str(out), "--config", str(config)]) == 0
        month_dir = out / "2025-03"  # без --month берётся последний месяц данных
        for name, schema in panel.OUTPUT_SCHEMAS.items():
            table = pd.read_parquet(month_dir / f"{name}.parquet")
            assert list(table.columns) == list(schema)

    def test_run_py_builds_the_same_panel(self, data_dir, config, tmp_path):
        panel.main(["--data", str(data_dir), "--out", str(tmp_path / "a"), "--month", "2025-02", "--config", str(config)])
        pipeline.main(["--month", "2025-02", "--data", str(data_dir), "--out", str(tmp_path / "b"), "--config", str(config)])
        for name in panel.OUTPUT_SCHEMAS:
            pd.testing.assert_frame_equal(
                pd.read_parquet(tmp_path / "a" / "2025-02" / f"{name}.parquet"),
                pd.read_parquet(tmp_path / "b" / "2025-02" / f"{name}.parquet"),
            )

    def test_bad_month(self, data_dir, config, tmp_path, caplog):
        assert panel.main(["--data", str(data_dir), "--out", str(tmp_path), "--month", "март", "--config", str(config)]) == 1
        assert "укажите месяц в формате ГГГГ-ММ" in caplog.text


# ---------------------------------------------------------------------------
# Настоящие данные
# ---------------------------------------------------------------------------

@pytest.mark.real_data
def test_real_data_panel(real_data_dir, settings):
    raw, _ = quality.clean(loaders.load_all(real_data_dir, settings, loaders.default_cache_dir(settings)), settings)
    tables = panel.build_from_raw(raw, settings)
    p = tables["panel"]
    assert p["month"].drop_duplicates().sort_values().tolist() == list(pd.date_range("2025-01-01", "2026-03-01", freq="MS"))
    assert (p.groupby("ls").size() == 15).all()
    assert np.allclose(p["dz"], p["saldo_start"].clip(lower=0), equal_nan=True)
    # Оплаты 03 и «Оплачено» ОСВ почти всегда совпадают (DATA-8 — подробная сверка).
    assert ((p["paid"] - p["paid_payments"]).abs() <= 1).mean() > 0.99
    assert tables["measure_events"]["n_raw"].sum() == len(raw["events"])
