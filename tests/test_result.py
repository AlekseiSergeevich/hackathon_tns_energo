"""«Результат» меры (P-03, P-04, DATA-6): оплата ≥ X % долга в окне после меры.

X — result.share, окно — result.window (month_end / next_month_end / days) и result.days (N).
В примерах ЛС 1 с долгом 100 ₽ на 01-е каждого месяца, данные 2025-01…2025-03 (конец данных — 31.03.2025).
"""
from __future__ import annotations

import logging

import pandas as pd
import pytest
import yaml

from tns.data import loaders, panel, quality
from tests.conftest import LIMITS_YAML, make_events, make_osv, make_payments
from tests.test_panel import build

T = pd.Timestamp


@pytest.fixture
def osv():
    return make_osv({1: [(100, 0, 0)] * 3, 2: [(0, 0, 0)] * 3})


def events_log(osv, payments, events, settings):
    return build(osv, make_payments(payments), make_events(events), settings=settings)["measure_events"]


def assert_result_consistent(log: pd.DataFrame, share: float, data_end: pd.Timestamp) -> None:
    """Результат = оплачено в окне ≥ X × долга; пусто, если долга нет или окно не закрылось."""
    known = (log["dz_at_event"] > 0) & (log["window_end"] <= data_end)
    expected = log["paid_in_window"] >= (share * log["dz_at_event"]).round(2)
    assert log.loc[known, "result_30pct"].notna().all()
    assert (log.loc[known, "result_30pct"] == expected[known]).all()
    assert log.loc[~known, "result_30pct"].isna().all()


class TestShare:
    @pytest.mark.parametrize("share, result", [(0.30, True), (0.40, True), (0.41, False), (1.0, False)])
    def test_share_from_settings(self, osv, settings, share, result):
        settings["result"]["share"] = share
        log = events_log(osv, [(1, "2025-02-10", 40, 5)], [(1, "sms", "2025-02-01")], settings)
        assert log["paid_in_window"].tolist() == [40]
        assert log["result_30pct"].tolist() == [result]

    def test_ml_target_uses_the_same_share(self, osv, settings):
        settings["result"]["share"] = 0.5
        tables = build(osv, make_payments([(1, "2025-02-10", 40, 5)]), settings=settings)
        feb = tables["panel"].set_index(["ls", "month"]).loc[(1, T("2025-02-01"))]
        assert feb["pay30_m"] == False  # noqa: E712  40 < 0.5 × 100


class TestDaysWindow:
    def test_day_precision_window_is_n_days_from_the_measure(self, osv, settings):
        # Ограничение 20.02, N = 10: окно 20.02…01.03 включительно.
        settings["result"].update(window="days", days=10)
        payments = [(1, "2025-02-19", 500, 5), (1, "2025-03-01", 30, 5), (1, "2025-03-02", 500, 5)]
        log = events_log(osv, payments, [(1, "restriction", "2025-02-20")], settings)
        assert log["window_end"].tolist() == [T("2025-03-01")]
        assert log["paid_in_window"].tolist() == [30] and log["result_30pct"].tolist() == [True]

    def test_month_precision_window_starts_on_the_1st(self, osv, settings):
        # СМС: известен только месяц, окно с 01.02. N = 1 — только сам день 01.02.
        settings["result"].update(window="days", days=1)
        payments = [(1, "2025-02-01", 20, 5), (1, "2025-02-02", 500, 5)]
        log = events_log(osv, payments, [(1, "sms", "2025-02-01")], settings)
        assert log["window_end"].tolist() == [T("2025-02-01")]
        assert log["paid_in_window"].tolist() == [20] and log["result_30pct"].tolist() == [False]

    def test_month_precision_date_not_on_the_1st(self, osv, settings):
        # Выезд — точность «месяц», хотя в файле 15.02. Окно N = 10: с 01.02 по 10.02, оплата 05.02 входит.
        settings["result"].update(window="days", days=10)
        log = events_log(osv, [(1, "2025-02-05", 30, 5)], [(1, "visit", "2025-02-15")], settings)
        assert log["window_end"].tolist() == [T("2025-02-10")]
        assert log["paid_in_window"].tolist() == [30] and log["result_30pct"].tolist() == [True]

    def test_window_crosses_into_next_month(self, osv, settings):
        settings["result"].update(window="days", days=30)
        log = events_log(osv, [(1, "2025-03-05", 30, 5)], [(1, "claim", "2025-02-01")], settings)
        assert log["window_end"].tolist() == [T("2025-03-02")]
        assert log["paid_in_window"].tolist() == [0] and log["result_30pct"].tolist() == [False]


class TestOpenWindow:
    """Окно кончается после конца данных (31.03.2025): результат ещё неизвестен, а не «нет»."""

    @pytest.mark.parametrize("window, days, march_known", [
        ("month_end", None, True),         # окно до 31.03 — закрыто
        ("next_month_end", None, False),   # до 30.04 — открыто
        ("days", 31, True),                # 01.03 + 30 дней = 31.03 — закрыто
        ("days", 32, False),               # до 01.04 — открыто
    ])
    def test_march_measure(self, osv, settings, window, days, march_known):
        settings["result"].update(window=window, days=days)
        events = [(1, "sms", "2025-02-01"), (1, "sms", "2025-03-01")]
        log = events_log(osv, [(1, "2025-03-10", 50, 5)], events, settings).set_index("month")
        assert log.loc[T("2025-03-01"), "paid_in_window"] == 50
        assert pd.notna(log.loc[T("2025-03-01"), "result_30pct"]) == march_known
        assert pd.notna(log.loc[T("2025-02-01"), "result_30pct"])  # февральское окно закрыто всегда

    def test_open_windows_are_reported(self, osv, settings, caplog):
        settings["result"]["window"] = "next_month_end"
        with caplog.at_level(logging.INFO):
            events_log(osv, [], [(1, "sms", "2025-03-01")], settings)
        assert "у 1 мер окно результата кончается после 31.03.2025" in caplog.text


class TestSettingsErrors:
    @pytest.mark.parametrize("change, message", [
        ({"share": 0}, "result.share — доля долга от 0 до 1"),
        ({"share": 30}, "result.share — доля долга от 0 до 1"),
        ({"share": "30%"}, "result.share — доля долга от 0 до 1"),
        ({"share": None}, "result.share — доля долга от 0 до 1"),
        ({"window": "week"}, "Неизвестное значение result.window: «week». Можно: month_end, next_month_end, days"),
        ({"window": "days", "days": 0}, "result.days — целое число дней не меньше 1"),
        ({"window": "days", "days": 2.5}, "result.days — целое число дней не меньше 1"),
        ({"window": "days", "days": None}, "result.days — целое число дней не меньше 1"),
    ])
    def test_bad_values_stop_the_build(self, osv, settings, change, message):
        settings["result"].update(change)
        with pytest.raises(ValueError, match=message):
            build(osv, settings=settings)

    def test_days_is_ignored_for_month_windows(self, settings):
        settings["result"].update(window="month_end", days=None)
        assert panel.check_result_settings(settings) == (0.30, "month_end", None)


def test_config_file_changes_the_result(data_dir, settings, tmp_path):
    """X и N меняются в файле настроек — команда пересчитывает результат без правки кода."""
    fallback = tmp_path / "limits.yaml"
    fallback.write_text(yaml.safe_dump(LIMITS_YAML), encoding="utf-8")
    settings["paths"] = {**settings["paths"], "cache_dir": str(tmp_path / "cache"), "limits_fallback": str(fallback)}
    logs = {}
    for name, result in {"default": {}, "custom": {"share": 0.5, "window": "days", "days": 5}}.items():
        config = tmp_path / f"{name}.yaml"
        config.write_text(yaml.safe_dump({**settings, "result": {**settings["result"], **result}}, allow_unicode=True), encoding="utf-8")
        assert panel.main(["--data", str(data_dir), "--out", str(tmp_path / name), "--month", "2025-03", "--config", str(config)]) == 0
        logs[name] = pd.read_parquet(tmp_path / name / "2025-03" / "measure_events.parquet")

    custom = logs["custom"]
    starts = custom["event_date"].where(custom["date_precision"] == "day", custom["month"])
    assert (custom["window_end"] == starts + pd.Timedelta(days=4)).all()
    assert not custom["window_end"].equals(logs["default"]["window_end"])
    data_end = custom["month"].max() + pd.offsets.MonthEnd(1)
    assert_result_consistent(custom, 0.5, data_end)
    assert_result_consistent(logs["default"], 0.30, data_end)


@pytest.mark.real_data
def test_real_data_result_for_every_measure(real_data_dir, settings):
    raw, _ = quality.clean(loaders.load_all(real_data_dir, settings, loaders.default_cache_dir(settings)), settings)
    for result in [{}, {"share": 0.5, "window": "days", "days": 30}]:
        settings["result"].update(result)
        share = settings["result"]["share"]
        tables = panel.build_from_raw(raw, settings)
        log = tables["measure_events"]
        # Каждая мера истории — в журнале (повторы месяца свёрнуты, n_raw хранит исходное число строк).
        assert log["n_raw"].sum() == len(raw["events"])
        data_end = tables["panel"]["month"].max() + pd.offsets.MonthEnd(1)
        assert_result_consistent(log, share, data_end)
        # Пусто — только если долга нет или окно ещё не закрылось.
        closed_with_debt = (log["dz_at_event"] > 0) & (log["window_end"] <= data_end)
        assert closed_with_debt.mean() > 0.9
