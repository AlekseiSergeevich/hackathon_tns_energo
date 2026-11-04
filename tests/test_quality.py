"""Тесты очистки и отчёта о качестве данных (DATA-3, tns/data/quality.py)."""
from __future__ import annotations

import logging

import pandas as pd
import pytest
import yaml

from tns import pipeline
from tns.data import loaders, panel, quality
from tns.data.validator import WARNING, Problem
from tests.conftest import make_events, make_info, make_osv, make_payments

AS_OF = pd.Timestamp("2025-06-30")
CHECKS = [f"Q-{i:02d}" for i in range(1, 18)]


@pytest.fixture
def raw():
    """Два активных ЛС (1, 2), один без движения (3) и один без движения, но с мерой (4)."""
    osv = make_osv({
        1: [(100, 50, 0), (150, 50, 50), (150, 50, 0)],
        2: [(-20, 30, 0), (10, 30, 30), (10, 30, 0)],
        3: [(0, 0, 0), (0, 0, 0), (0, 0, 0)],
        4: [(0, 0, 0), (0, 0, 0), (0, 0, 0)],
    })
    return {
        "info": make_info([1, 2, 3, 4]),
        "osv": osv,
        "payments": make_payments([(1, "2025-02-10", 50, 5), (2, "2025-02-11", 30, 4)]),
        "events": make_events([(1, "sms", "2025-02-01"), (4, "claim", "2025-02-01")]),
        "limits": loaders._limits_frame(dict.fromkeys(loaders.MEASURE_FILES, 10), []),
    }


@pytest.fixture
def settings():
    return loaders.load_settings()


def run(raw, settings, **kwargs):
    cleaned, report = quality.clean(raw, settings, as_of=AS_OF, **kwargs)
    return cleaned, report, report.summary.set_index("code")


def count(summary, code) -> int:
    return int(summary.loc[code, "count"])


class TestReportShape:
    def test_every_check_is_listed_even_if_nothing_found(self, raw, settings):
        _, report, _ = run(raw, settings)
        assert [c for c in report.summary["code"] if c != "Q-00"] == CHECKS
        assert report.summary["check"].str.len().gt(0).all()

    def test_types(self, raw, settings):
        _, report, _ = run(raw, settings)
        assert {c: str(t) for c, t in report.summary.dtypes.items()} == {
            "code": "string", "check": "string", "source": "string", "count": "Int64", "total": "Int64",
            "share": "float64", "action": "string", "details": "string",
        }
        assert str(report.examples["ls"].dtype) == "Int64"

    def test_input_is_not_changed(self, raw, settings):
        before = {name: table.copy() for name, table in raw.items()}
        run(raw, settings)
        for name, table in raw.items():
            pd.testing.assert_frame_equal(table, before[name])

    def test_as_of_from_settings_and_today(self, raw, settings):
        settings["quality"]["as_of"] = "2025-03-15"
        assert quality.clean(raw, settings)[1].as_of == pd.Timestamp("2025-03-15")
        settings["quality"]["as_of"] = None
        assert quality.clean(raw, settings)[1].as_of == pd.Timestamp.today().normalize()


class TestDuplicates:
    def test_duplicate_ls_from_loader_shown_as_dropped(self, raw, settings):
        raw["info"].attrs["problems"] = [Problem(
            WARNING, loaders.INFO_FILE, "В файле «01…» 2 повтора ЛС, оставлена первая строка.",
            column="ЛС", count=2, total=6, blocking=False,
        )]
        _, report, _ = run(raw, settings)
        row = report.summary[report.summary["code"] == "Q-00"].iloc[0]
        assert (row["check"], row["source"], row["count"], row["total"]) == ("Повторы ЛС", "01", 2, 6)
        assert row["action"] == "убрано: оставлена первая строка"

    def test_duplicate_ls_end_to_end(self, data_dir, settings):
        from tests.conftest import info_row, write_info
        write_info(data_dir / loaders.INFO_FILE, [info_row(10), info_row(10), info_row(20), info_row(30)])
        raw = loaders.load_all(data_dir, settings)
        cleaned, report = quality.clean(raw, settings, as_of=AS_OF)
        assert cleaned["info"]["ls"].is_unique
        assert "Повторы ЛС" in set(report.summary["check"])

    def test_repeated_measures_in_a_month(self, raw, settings):
        raw["events"] = make_events([(1, "sms", "2025-02-01")] * 3 + [(1, "visit", "2025-02-03"), (1, "visit", "2025-02-20")])
        _, report, summary = run(raw, settings)
        assert count(summary, "Q-15") == 3
        assert summary.loc["Q-15", "details"] == "sms: 2; visit: 1"
        example = report.examples[report.examples["code"] == "Q-15"].iloc[0]
        assert (example["ls"], example["when"], example["value"]) == (1, "2025-02", "sms × 3")

    def test_identical_payments_kept(self, raw, settings):
        raw["payments"] = make_payments([(1, "2025-02-10", 50, 5)] * 2)
        cleaned, _, summary = run(raw, settings)
        assert count(summary, "Q-16") == 1 and len(cleaned["payments"]) == 2


class TestNegativeDebt:
    def test_negative_saldo_reported_and_kept(self, raw, settings):
        cleaned, report, summary = run(raw, settings)
        assert count(summary, "Q-05") == 1 and summary.loc["Q-05", "total"] == 12
        assert "1 ЛС; в долг не входит: dz = max(0, saldo_start)" == summary.loc["Q-05", "details"]
        assert cleaned["osv"]["saldo_start"].min() == -20
        example = report.examples[report.examples["code"] == "Q-05"].iloc[0]
        assert (example["ls"], example["when"], example["value"]) == (2, "2025-01", "-20,00")

    def test_panel_dz_is_zero_for_overpayment(self, raw, settings):
        cleaned, _, _ = run(raw, settings)
        built = panel.build_from_raw(cleaned, settings)["panel"]
        row = built[(built["ls"] == 2) & (built["month"] == "2025-01-01")].iloc[0]
        assert (row["saldo_start"], row["dz"]) == (-20, 0)

    def test_negative_accrual_and_reversal(self, raw, settings):
        raw["osv"].loc[0, "accrual"] = -5
        raw["payments"] = make_payments([(1, "2025-02-10", -50, 5), (1, "2025-02-11", 50, 5)])
        _, _, summary = run(raw, settings)
        assert count(summary, "Q-06") == 1 and count(summary, "Q-07") == 1


class TestSaldoMismatch:
    def test_mismatch_found_with_example(self, raw, settings):
        _, report, summary = run(raw, settings)
        # ЛС 1: 150 + 50 − 50 = 150 — сходится; ЛС 1, январь: 100 + 50 − 0 = 150 — сходится.
        # ЛС 2, январь: −20 + 30 − 0 = 10 — сходится; ЛС 2, февраль: 10 + 30 − 30 = 10 — сходится.
        assert count(summary, "Q-08") == 0
        raw["osv"].loc[(raw["osv"]["ls"] == 1) & (raw["osv"]["month"] == "2025-03-01"), "saldo_start"] = 999
        _, report, summary = run(raw, settings)
        assert count(summary, "Q-08") == 1
        example = report.examples[report.examples["code"] == "Q-08"].iloc[0]
        assert (example["ls"], example["when"], example["value"]) == (1, "2025-03", "ожидалось 150,00, в файле 999,00")

    def test_tolerance(self, raw, settings):
        raw["osv"].loc[(raw["osv"]["ls"] == 1) & (raw["osv"]["month"] == "2025-03-01"), "saldo_start"] = 150.004
        assert count(run(raw, settings)[2], "Q-08") == 0
        settings["quality"]["reconcile_tolerance"] = 0.001
        assert count(run(raw, settings)[2], "Q-08") == 1


class TestFutureDates:
    def test_future_payments_dropped(self, raw, settings):
        raw["payments"] = make_payments([(1, "2025-02-10", 50, 5), (1, "2025-07-01", 70, 5)])
        cleaned, report, summary = run(raw, settings)
        assert count(summary, "Q-09") == 1 and summary.loc["Q-09", "action"] == "убрано"
        assert cleaned["payments"]["payment_date"].max() == pd.Timestamp("2025-02-10")
        assert "позже 30.06.2025" in summary.loc["Q-09", "details"]
        example = report.examples[report.examples["code"] == "Q-09"].iloc[0]
        assert (example["ls"], example["when"]) == (1, "2025-07-01")

    def test_future_events_dropped(self, raw, settings):
        raw["events"] = make_events([(1, "sms", "2025-02-01"), (1, "restriction", "2026-01-15")])
        cleaned, _, summary = run(raw, settings)
        assert count(summary, "Q-10") == 1
        assert cleaned["events"]["measure"].tolist() == ["sms"]

    def test_as_of_day_itself_is_not_future(self, raw, settings):
        raw["payments"] = make_payments([(1, "2025-06-30", 50, 5)])
        assert count(run(raw, settings)[2], "Q-09") == 0

    def test_dates_outside_osv_period_kept(self, raw, settings):
        raw["payments"] = make_payments([(1, "2024-12-30", 50, 5), (1, "2025-04-02", 50, 5), (1, "2025-03-31", 50, 5)])
        raw["events"] = make_events([(1, "sms", "2025-05-01")])
        cleaned, _, summary = run(raw, settings)
        assert count(summary, "Q-11") == 2 and count(summary, "Q-12") == 1
        assert "период ОСВ 2025-01…2025-03" in summary.loc["Q-11", "details"]
        assert len(cleaned["payments"]) == 3 and len(cleaned["events"]) == 1


class TestInactive:
    def test_inactive_ls_dropped(self, raw, settings):
        cleaned, report, summary = run(raw, settings)
        assert count(summary, "Q-13") == 1 and summary.loc["Q-13", "total"] == 4
        assert 3 not in set(cleaned["osv"]["ls"]) and 3 not in set(cleaned["info"]["ls"])
        assert report.examples.loc[report.examples["code"] == "Q-13", "ls"].tolist() == [3]

    def test_inactive_with_measures_kept_and_reported(self, raw, settings):
        cleaned, report, summary = run(raw, settings)
        assert count(summary, "Q-14") == 1
        assert 4 in set(cleaned["osv"]["ls"])
        example = report.examples[report.examples["code"] == "Q-14"].iloc[0]
        assert (example["ls"], example["value"]) == (4, "claim")

    def test_ls_with_payments_is_not_inactive(self, raw, settings):
        raw["payments"] = make_payments([(3, "2025-02-10", 0.0, 5)])
        assert count(run(raw, settings)[2], "Q-13") == 0

    def test_setting_keeps_inactive(self, raw, settings):
        settings["quality"]["drop_inactive_ls"] = False
        cleaned, _, summary = run(raw, settings)
        assert count(summary, "Q-13") == 1 and summary.loc["Q-13", "action"] == "оставлено, только в отчёте"
        assert 3 in set(cleaned["osv"]["ls"])

    def test_attrs_survive_cleaning(self, raw, settings):
        raw["osv"].attrs["problems"] = []
        raw["osv"].attrs["cached"] = True
        cleaned, _, _ = run(raw, settings)
        assert cleaned["osv"].attrs == {"problems": [], "cached": True}


class TestMissingLs:
    def test_ls_only_in_one_file(self, raw, settings):
        raw["info"] = make_info([1, 2, 3, 4, 77])
        raw["osv"] = pd.concat([raw["osv"], make_osv({88: [(5, 5, 0), (10, 5, 0), (15, 5, 0)]})], ignore_index=True)
        raw["payments"] = make_payments([(1, "2025-02-10", 50, 5), (99, "2025-02-11", 30, 4)])
        raw["events"] = make_events([(1, "sms", "2025-02-01"), (98, "sms", "2025-02-01")])
        _, report, summary = run(raw, settings)
        assert [count(summary, c) for c in ("Q-01", "Q-02", "Q-03", "Q-04")] == [1, 1, 1, 1]
        by_code = report.examples.groupby("code")["ls"].first().to_dict()
        assert {k: by_code[k] for k in ("Q-01", "Q-02", "Q-03", "Q-04")} == {"Q-01": 77, "Q-02": 88, "Q-03": 99, "Q-04": 98}

    def test_null_payment_method(self, raw, settings):
        raw["payments"] = make_payments([(1, "2025-02-10", 50, None), (2, "2025-02-11", 30, 4)])
        assert count(run(raw, settings)[2], "Q-17") == 1


class TestExcel:
    def test_report_file(self, raw, settings, tmp_path):
        _, report, _ = run(raw, settings)
        path = quality.write_report(report, tmp_path / "2025-03", {"Месяц плана": "2025-03"})
        assert path == tmp_path / "2025-03" / "quality_report.xlsx"
        sheets = pd.read_excel(path, sheet_name=None)
        assert list(sheets) == ["Сводка", "Примеры", "Запуск"]
        assert list(sheets["Сводка"].columns) == ["Код", "Проверка", "Источник", "Найдено", "Из", "Доля", "Что сделано", "Подробности"]
        assert sheets["Сводка"]["Код"].tolist()[-17:] == CHECKS
        assert dict(zip(sheets["Запуск"]["Параметр"], sheets["Запуск"]["Значение"]))["Месяц плана"] == "2025-03"

    def test_examples_are_limited(self, raw, settings):
        settings["quality"]["examples"] = 2
        raw["payments"] = make_payments([(1, f"2025-07-{d:02d}", 1, 5) for d in range(1, 10)])
        _, report, summary = run(raw, settings)
        assert count(summary, "Q-09") == 9
        assert (report.examples["code"] == "Q-09").sum() == 2

    def test_file_open_in_excel(self, raw, settings, tmp_path, monkeypatch):
        _, report, _ = run(raw, settings)

        def locked(*args, **kwargs):
            raise PermissionError("locked")
        monkeypatch.setattr(pd, "ExcelWriter", locked)
        with pytest.raises(OSError, match="файл открыт в Excel\\? Закройте его"):
            quality.write_report(report, tmp_path, {})


class TestEveryRun:
    def config(self, settings, tmp_path):
        from tests.conftest import LIMITS_YAML
        fallback = tmp_path / "limits.yaml"
        fallback.write_text(yaml.safe_dump(LIMITS_YAML), encoding="utf-8")
        settings["paths"] = {**settings["paths"], "cache_dir": str(tmp_path / "cache"), "limits_fallback": str(fallback)}
        path = tmp_path / "settings.yaml"
        path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")
        return path

    def test_run_py_writes_report_each_time(self, data_dir, settings, tmp_path, caplog):
        config = self.config(settings, tmp_path)
        args = ["--month", "2025-02", "--data", str(data_dir), "--out", str(tmp_path / "out"), "--config", str(config)]
        report = tmp_path / "out" / "2025-02" / "quality_report.xlsx"
        with caplog.at_level(logging.INFO):
            assert pipeline.main(args) == 0
        first = report.stat().st_mtime_ns
        assert "Отчёт о качестве данных:" in caplog.text
        assert pipeline.main(args) == 0
        assert report.stat().st_mtime_ns >= first
        summary = pd.read_parquet(tmp_path / "out" / "2025-02" / "quality.parquet")
        assert [c for c in summary["code"] if c != "Q-00"] == CHECKS

    def test_panel_cli_writes_report(self, data_dir, settings, tmp_path):
        config = self.config(settings, tmp_path)
        assert panel.main(["--data", str(data_dir), "--out", str(tmp_path / "out"), "--month", "2025-02",
                           "--config", str(config), "--no-cache"]) == 0
        assert (tmp_path / "out" / "2025-02" / "quality_report.xlsx").exists()


@pytest.mark.real_data
def test_real_data(real_data_dir, settings):
    raw = loaders.load_all(real_data_dir, settings, loaders.default_cache_dir(settings))
    cleaned, report = quality.clean(raw, settings)
    summary = report.summary.set_index("code")
    assert int(summary.loc["Q-13", "count"]) > 10_000
    assert int(summary.loc["Q-05", "count"]) > 0 and int(summary.loc["Q-09", "count"]) == 0
    assert cleaned["osv"]["ls"].nunique() == raw["osv"]["ls"].nunique() - int(summary.loc["Q-13", "count"])
