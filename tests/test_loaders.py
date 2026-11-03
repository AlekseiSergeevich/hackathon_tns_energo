"""Тесты загрузчиков tns/data/loaders.py (DATA-1): по тесту-группе на каждый из файлов 01–14."""
from __future__ import annotations

import logging
import os
from datetime import datetime

import pandas as pd
import pytest
import yaml

from tns.data import loaders
from tns.data.validator import ValidationError
from tests.conftest import (
    INFO_HEADER, LIMITS_YAML, copy_without, default_osv_blocks, info_row,
    write_info, write_limits, write_measure, write_osv, write_payments, write_xlsx,
)


def lost(table: pd.DataFrame, column: str) -> int:
    """Сколько значений колонки не прочиталось — по предупреждениям загрузчика."""
    return sum(p.count for p in table.attrs["problems"] if p.column == column)


def messages(table: pd.DataFrame) -> str:
    return "\n".join(p.message for p in table.attrs["problems"])


# ---------------------------------------------------------------------------
# Файл 01 — анкета
# ---------------------------------------------------------------------------

class TestInfo:
    def test_types_and_values(self, data_dir):
        info = loaders.read_info(data_dir)
        loaders.check_schema("info", info)
        by_ls = info.set_index("ls")
        assert by_ls.loc[1, "has_phone"] is True or by_ls.loc[1, "has_phone"] == True  # noqa: E712
        assert by_ls.loc[1, "has_ebill"] == True and by_ls.loc[1, "has_benefits"] == False  # noqa: E712
        assert by_ls.loc[2, "remote_disconnect"] == True and by_ls.loc[2, "has_benefits"] == True  # noqa: E712
        assert by_ls.loc[3, "gasified"] == True  # noqa: E712
        assert by_ls.loc[1, "address_guid"] == "guid-1"
        assert list(by_ls["house_type"].astype(object)[[1, 2]]) == ["МКД", "ЧД"]

    def test_house_type_unknown_when_nothing_marked(self, data_dir):
        info = loaders.read_info(data_dir).set_index("ls")
        assert pd.isna(info.loc[3, "house_type"])

    def test_house_type_first_marked_wins(self, data_dir, caplog):
        row = info_row(7, house="ЧД")
        row[INFO_HEADER.index("МКД")] = "Да"
        write_info(data_dir / loaders.INFO_FILE, [row])
        with caplog.at_level(logging.WARNING):
            info = loaders.read_info(data_dir)
        assert info["house_type"].iloc[0] == "ЧД"
        assert "несколько типов дома" in caplog.text

    def test_unknown_flag_value_becomes_na_and_is_reported(self, data_dir, caplog):
        write_info(data_dir / loaders.INFO_FILE, [info_row(1, Наличие_телефона="Возможно"), info_row(2)])
        with caplog.at_level(logging.WARNING):
            info = loaders.read_info(data_dir)
        assert pd.isna(info.loc[0, "has_phone"])
        assert "Наличие телефона" in caplog.text
        assert lost(info, "Наличие телефона") == 1
        assert "строка 2 — «Возможно»" in messages(info)

    def test_duplicate_ls_keeps_first(self, data_dir, caplog):
        write_info(data_dir / loaders.INFO_FILE, [info_row(1, Наличие_телефона="Да"), info_row(1, Наличие_телефона="Нет")])
        with caplog.at_level(logging.WARNING):
            info = loaders.read_info(data_dir)
        assert len(info) == 1 and info.loc[0, "has_phone"] == True  # noqa: E712
        assert "1 повтор ЛС" in messages(info)

    def test_row_without_ls_is_dropped(self, data_dir):
        write_info(data_dir / loaders.INFO_FILE, [info_row(1), info_row(None)])
        info = loaders.read_info(data_dir)
        assert info["ls"].tolist() == [1]
        assert info["ls"].dtype == "int64"

    def test_missing_column_message(self, data_dir):
        header = [c for c in INFO_HEADER if c != "Наличие льгот"]
        write_xlsx(data_dir / loaders.INFO_FILE, [header, ["1"] * len(header)])
        with pytest.raises(ValueError, match="в файле «01 .*нет колонки «Наличие льгот»".replace("в файле", "файле")):
            loaders.read_info(data_dir)

    def test_missing_file_message(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Не найден входной файл"):
            loaders.read_info(tmp_path)


# ---------------------------------------------------------------------------
# Файл 02 — ОСВ
# ---------------------------------------------------------------------------

class TestOsv:
    def test_long_format_and_types(self, data_dir, settings):
        osv = loaders.read_osv(data_dir, settings)
        loaders.check_schema("osv", osv)
        assert len(osv) == 3 * 3
        assert osv["month"].drop_duplicates().tolist() == list(pd.date_range("2025-01-01", periods=3, freq="MS"))
        assert (osv["month"].dt.day == 1).all()
        row = osv[(osv["ls"] == 20) & (osv["month"] == "2025-02-01")].iloc[0]
        assert (row["saldo_start"], row["accrual"], row["paid"]) == (600, 100, 100)

    def test_typo_in_header_is_understood(self, data_dir, settings):
        osv = loaders.read_osv(data_dir, settings)
        assert osv["paid"].notna().all()

    def test_sorted_by_ls_and_month(self, data_dir, settings):
        osv = loaders.read_osv(data_dir, settings)
        assert osv.equals(osv.sort_values(["ls", "month"], ignore_index=True))

    def test_duplicate_month_shift_next(self, data_dir, settings, caplog):
        with caplog.at_level(logging.WARNING):
            osv = loaders.read_osv(data_dir, settings)
        march = osv[(osv["ls"] == 20) & (osv["month"] == "2025-03-01")].iloc[0]
        assert march["saldo_start"] == 600
        assert "повторяется в шапке" in caplog.text and "100,0 %" in caplog.text

    def test_duplicate_month_drop(self, data_dir, settings):
        settings["osv"]["duplicate_month"] = "drop"
        osv = loaders.read_osv(data_dir, settings)
        assert osv["month"].max() == pd.Timestamp("2025-02-01")
        assert len(osv) == 3 * 2

    def test_duplicate_month_error(self, data_dir, settings):
        settings["osv"]["duplicate_month"] = "error"
        with pytest.raises(ValueError, match="повторяется в шапке"):
            loaders.read_osv(data_dir, settings)

    def test_unknown_policy(self, data_dir, settings):
        settings["osv"]["duplicate_month"] = "maybe"
        with pytest.raises(ValueError, match="osv.duplicate_month"):
            loaders.read_osv(data_dir, settings)

    def test_shift_blocked_when_next_month_exists(self, data_dir, settings):
        blocks, ls_values = default_osv_blocks()
        write_osv(data_dir / loaders.OSV_FILE, [blocks[0], blocks[1], blocks[1], (datetime(2025, 3, 1), blocks[2][1])], ls_values)
        with pytest.raises(ValueError, match="сдвинуть блок нельзя"):
            loaders.read_osv(data_dir, settings)

    def test_gap_between_months(self, data_dir, settings):
        blocks, ls_values = default_osv_blocks()
        write_osv(data_dir / loaders.OSV_FILE, [blocks[0], (datetime(2025, 4, 1), blocks[1][1])], ls_values)
        with pytest.raises(ValueError, match="пропущены месяцы"):
            loaders.read_osv(data_dir, settings)

    def test_missing_metric(self, data_dir, settings):
        blocks, ls_values = default_osv_blocks()
        write_osv(data_dir / loaders.OSV_FILE, blocks[:1], ls_values, metrics=["СЗ на начало", "Начислено", "Что-то"])
        with pytest.raises(ValueError, match="для месяца 2025-01 нет колонки «Оплачено»"):
            loaders.read_osv(data_dir, settings)

    def test_rows_without_ls_dropped_and_non_numbers_reported(self, data_dir, settings):
        blocks, _ = default_osv_blocks()
        blocks = [(month, values[:2] + [["abc", 1, 1]]) for month, values in blocks]
        write_osv(data_dir / loaders.OSV_FILE, blocks, [10, 20, None])
        osv = loaders.read_osv(data_dir, settings)
        assert sorted(osv["ls"].unique()) == [10, 20]
        assert lost(osv, "ЛС") == 1

    def test_unreadable_number_is_nan_and_reported(self, data_dir, settings):
        blocks, ls_values = default_osv_blocks()
        blocks[0][1][0][0] = "н/д"
        write_osv(data_dir / loaders.OSV_FILE, blocks, ls_values)
        osv = loaders.read_osv(data_dir, settings)
        assert osv.loc[(osv["ls"] == 10) & (osv["month"] == "2025-01-01"), "saldo_start"].isna().all()
        assert lost(osv, "СЗ на начало") == 1 and "строка 3 — «н/д»" in messages(osv)


# ---------------------------------------------------------------------------
# Файл 03 — оплаты
# ---------------------------------------------------------------------------

class TestPayments:
    def test_types_and_money(self, data_dir):
        payments = loaders.read_payments(data_dir)
        loaders.check_schema("payments", payments)
        assert payments["amount"].tolist() == [100.0, 1240.5, 1240.5, -30.5]
        assert payments["payment_date"].iloc[0] == pd.Timestamp("2025-01-15")

    def test_method_code_and_null(self, data_dir):
        payments = loaders.read_payments(data_dir)
        assert payments["payment_method"].astype(object).where(payments["payment_method"].notna(), None).tolist() == [5, None, None, 0]

    def test_duplicates_and_reversals_kept(self, data_dir):
        payments = loaders.read_payments(data_dir)
        assert payments.duplicated().sum() == 1
        assert (payments["amount"] < 0).sum() == 1

    def test_bad_rows_dropped_and_reported(self, data_dir, caplog):
        write_payments(data_dir / loaders.PAYMENTS_FILE, [
            "10;15.01.2025;100;5", "10;32.13.2025;100;5", "10;15.01.2025;много;5", "abc;15.01.2025;100;5",
        ])
        with caplog.at_level(logging.WARNING):
            payments = loaders.read_payments(data_dir)
        assert len(payments) == 1
        assert lost(payments, "Дата оплаты") == 1 and lost(payments, "Сумма") == 1
        assert lost(payments, "Номер") == 1

    def test_unknown_method_reported(self, data_dir):
        write_payments(data_dir / loaders.PAYMENTS_FILE, ["10;15.01.2025;100;карта"])
        payments = loaders.read_payments(data_dir)
        assert payments["payment_method"].isna().all()
        assert lost(payments, "Способ оплаты") == 1

    def test_missing_column_message(self, data_dir):
        (data_dir / loaders.PAYMENTS_FILE).write_text("Номер;Дата оплаты;Сумма\n10;15.01.2025;100\n", encoding="utf-8-sig")
        with pytest.raises(ValueError, match="03 Оплаты ХК.csv.*нет колонки «Способ оплаты»"):
            loaders.read_payments(data_dir)


# ---------------------------------------------------------------------------
# Файлы 04–13 — меры
# ---------------------------------------------------------------------------

class TestMeasureEvents:
    def test_all_ten_measures_in_one_table(self, data_dir):
        events = loaders.read_measure_events(data_dir)
        loaders.check_schema("events", events)
        assert len(events) == 10
        assert set(events["measure"]) == set(loaders.MEASURE_FILES)
        assert list(events["measure"].cat.categories) == list(loaders.MEASURE_FILES)

    @pytest.mark.parametrize("measure", list(loaders.MEASURE_FILES))
    def test_each_file_is_read(self, data_dir, measure):
        write_measure(
            data_dir / loaders.MEASURE_FILES[measure], measure,
            [[10, datetime(2025, 3, 1)], [11, datetime(2025, 4, 1)]],
        )
        events = loaders.read_measure_events(data_dir)
        mine = events[events["measure"] == measure]
        assert mine["ls"].tolist() == [10, 11]
        assert mine["event_date"].tolist() == [pd.Timestamp("2025-03-01"), pd.Timestamp("2025-04-01")]

    def test_date_precision(self, data_dir):
        events = loaders.read_measure_events(data_dir)
        precision = events.groupby("measure", observed=True)["date_precision"].first().astype(str)
        for measure, value in precision.items():
            assert value == ("month" if measure in loaders.MONTH_PRECISION_MEASURES else "day"), measure
        assert loaders.MONTH_PRECISION_MEASURES == {"autodial", "email", "sms", "claim", "visit"}

    def test_exact_duplicates_are_kept_for_n_raw(self, data_dir):
        write_measure(data_dir / loaders.MEASURE_FILES["autodial"], "autodial", [[10, datetime(2025, 1, 1)]] * 3)
        events = loaders.read_measure_events(data_dir)
        assert (events["measure"] == "autodial").sum() == 3

    def test_bad_rows_dropped_and_reported(self, data_dir):
        write_measure(data_dir / loaders.MEASURE_FILES["claim"], "claim", [
            [10, datetime(2025, 1, 1)], [11, "не дата"], [None, datetime(2025, 1, 1)],
        ])
        events = loaders.read_measure_events(data_dir)
        assert events[events["measure"] == "claim"]["ls"].tolist() == [10]
        assert {(p.file, p.column, p.count) for p in events.attrs["problems"]} == {
            ("08 Претензия ХК.xlsx", "ЛС", 1), ("08 Претензия ХК.xlsx", "Дата", 1),
        }

    def test_non_first_day_for_month_precision_warns(self, data_dir, caplog):
        write_measure(data_dir / loaders.MEASURE_FILES["sms"], "sms", [[10, datetime(2025, 1, 17)]])
        with caplog.at_level(logging.WARNING):
            loaders.read_measure_events(data_dir)
        assert "06 СМС ХК.xlsx" in caplog.text and "не на 1-е число" in caplog.text

    def test_missing_file_names_the_file(self, data_dir):
        (data_dir / loaders.MEASURE_FILES["sms"]).unlink()
        with pytest.raises(ValidationError, match="Не найден входной файл «06 СМС ХК.xlsx»"):
            loaders.read_measure_events(data_dir)

    def test_missing_column_names_file_and_column(self, data_dir):
        write_xlsx(data_dir / loaders.MEASURE_FILES["sms"], [["sms", None], ["ЛС", "Когда"], [10, datetime(2025, 1, 1)]])
        with pytest.raises(ValueError, match="06 СМС ХК.xlsx.*нет колонки «Дата»"):
            loaders.read_measure_events(data_dir)


# ---------------------------------------------------------------------------
# Файл 14 — лимиты
# ---------------------------------------------------------------------------

class TestLimits:
    def test_parsed_by_file_number(self, data_dir, settings):
        limits = loaders.read_limits(data_dir, settings)
        loaders.check_schema("limits", limits)
        values = limits.set_index("measure")["limit"]
        assert values["autodial"] == 8000 and values["court_result"] == 250 and values["restriction"] == 200

    def test_no_limit_text_is_na(self, data_dir, settings):
        values = loaders.read_limits(data_dir, settings).set_index("measure")["limit"]
        assert pd.isna(values["email"])

    def test_numbers_stored_as_text_or_float(self, data_dir, settings):
        rows = [[f"{n:02d} x", v] for n, v in zip(range(4, 14), ["8000", 2.0, "нет ограничений", 1, 2, 3, 4, 5, 6, 7])]
        write_limits(data_dir / loaders.LIMITS_FILE, rows)
        values = loaders.read_limits(data_dir, settings).set_index("measure")["limit"]
        assert values["autodial"] == 8000 and values["email"] == 2 and pd.isna(values["sms"])

    def test_fallback_when_file_missing(self, data_dir, settings, caplog, tmp_path):
        folder = copy_without(data_dir, loaders.LIMITS_FILE, tmp_path / "no14")
        with caplog.at_level(logging.WARNING):
            limits = loaders.read_limits(folder, settings)
        assert "Не найден файл" in caplog.text
        reserve = LIMITS_YAML["limits"]
        assert limits.set_index("measure")["limit"].astype(object).where(lambda s: s.notna(), None).to_dict() == reserve

    def test_warns_when_file_differs_from_fallback(self, data_dir, settings, caplog):
        reserve = yaml.safe_load(open(settings["paths"]["limits_fallback"], encoding="utf-8"))
        reserve["limits"]["sms"] = 1000
        yaml.safe_dump(reserve, open(settings["paths"]["limits_fallback"], "w", encoding="utf-8"))
        with caplog.at_level(logging.WARNING):
            limits = loaders.read_limits(data_dir, settings)
        assert "sms: 2150 против 1000" in caplog.text
        assert limits.set_index("measure").loc["sms", "limit"] == 2150

    def test_unknown_measure(self, data_dir, settings):
        write_limits(data_dir / loaders.LIMITS_FILE, [["99 Новая мера", 5]])
        with pytest.raises(ValueError, match="неизвестная мера «99 Новая мера»"):
            loaders.read_limits(data_dir, settings)

    def test_bad_value(self, data_dir, settings):
        rows = [[name, 1] for name in ("04 а", "05 б", "06 в", "07 г", "08 д", "09 е", "10 ж", "11 з", "12 и")]
        write_limits(data_dir / loaders.LIMITS_FILE, [*rows, ["13 к", "много"]])
        with pytest.raises(ValueError, match="лимит «много»"):
            loaders.read_limits(data_dir, settings)

    def test_missing_measure(self, data_dir, settings):
        write_limits(data_dir / loaders.LIMITS_FILE, [["04 Автодозвон ХК", 5]])
        with pytest.raises(ValueError, match="нет лимитов для мер"):
            loaders.read_limits(data_dir, settings)


# ---------------------------------------------------------------------------
# Типы по контракту docs/interfaces.md
# ---------------------------------------------------------------------------

class TestSchema:
    def test_contract_types(self, data_dir, settings):
        tables = loaders.load_all(data_dir, settings)
        expected = {
            "info": {"ls": "int64", "has_phone": "boolean", "has_ebill": "boolean", "has_benefits": "boolean",
                     "not_residing": "boolean", "remote_disconnect": "boolean", "gasified": "boolean",
                     "house_type": "category", "address_guid": "string"},
            "osv": {"ls": "int64", "month": "datetime64[ns]", "saldo_start": "float64", "accrual": "float64", "paid": "float64"},
            "payments": {"ls": "int64", "payment_date": "datetime64[ns]", "amount": "float64", "payment_method": "Int64"},
            "events": {"ls": "int64", "measure": "category", "event_date": "datetime64[ns]", "date_precision": "category"},
            "limits": {"measure": "category", "limit": "Int64"},
        }
        for name, columns in expected.items():
            assert {c: str(t) for c, t in tables[name].dtypes.items()} == columns, name
        assert list(tables["info"]["house_type"].cat.categories) == ["ЧД", "МКД", "Общежитие"]

    def test_ls_has_no_gaps(self, data_dir, settings):
        for name, table in loaders.load_all(data_dir, settings).items():
            if "ls" in table.columns:
                assert not table["ls"].isna().any(), name

    def test_wrong_type_is_reported(self, data_dir, settings):
        info = loaders.read_info(data_dir)
        info["ls"] = info["ls"].astype("float64")
        with pytest.raises(ValueError, match="«ls»: ожидался тип int64, получено float64"):
            loaders.check_schema("info", info)

    def test_missing_and_extra_columns_are_reported(self, data_dir):
        info = loaders.read_info(data_dir).drop(columns=["gasified"]).assign(лишняя=1)
        with pytest.raises(ValueError) as error:
            loaders.check_schema("info", info)
        assert "нет колонки «gasified»" in str(error.value) and "лишние колонки: лишняя" in str(error.value)

    def test_wrong_categories_are_reported(self, data_dir):
        info = loaders.read_info(data_dir)
        info["house_type"] = info["house_type"].cat.rename_categories(["a", "b", "c"])
        with pytest.raises(ValueError, match="house_type"):
            loaders.check_schema("info", info)


# ---------------------------------------------------------------------------
# load_all и кэш
# ---------------------------------------------------------------------------

class TestLoadAll:
    def test_one_call_reads_everything(self, data_dir, settings):
        tables = loaders.load_all(data_dir, settings)
        assert list(tables) == ["info", "osv", "payments", "events", "limits"]

    def test_missing_folder(self, tmp_path, settings):
        with pytest.raises(FileNotFoundError, match="Не найдена папка с данными"):
            loaders.load_all(tmp_path / "нет такой", settings)

    def test_no_cache_files_without_cache_dir(self, data_dir, settings, tmp_path):
        loaders.load_all(data_dir, settings)
        assert not (tmp_path / "cache").exists()

    def test_cache_gives_same_tables(self, data_dir, settings, tmp_path):
        cache = tmp_path / "cache"
        first = loaders.load_all(data_dir, settings, cache)
        second = loaders.load_all(data_dir, settings, cache)
        assert all(not first[n].attrs["cached"] for n in ("info", "osv", "payments", "events"))
        assert all(second[n].attrs["cached"] for n in ("info", "osv", "payments", "events"))
        for name in first:
            pd.testing.assert_frame_equal(first[name], second[name])

    def test_cache_keeps_problems(self, data_dir, settings, tmp_path):
        write_payments(data_dir / loaders.PAYMENTS_FILE, ["10;15.01.2025;100;5", "10;15.01.2025;много;5"])
        cache = tmp_path / "cache"
        first = loaders.load_all(data_dir, settings, cache)
        second = loaders.load_all(data_dir, settings, cache)
        assert second["payments"].attrs["cached"]
        assert second["payments"].attrs["problems"] == first["payments"].attrs["problems"]
        assert lost(second["payments"], "Сумма") == 1

    def test_cache_reset_when_source_file_changes(self, data_dir, settings, tmp_path):
        cache = tmp_path / "cache"
        loaders.load_all(data_dir, settings, cache)
        path = data_dir / loaders.PAYMENTS_FILE
        write_payments(path, ["10;15.01.2025;500;5"])
        os.utime(path, ns=(1, 1))
        tables = loaders.load_all(data_dir, settings, cache)
        assert not tables["payments"].attrs["cached"] and tables["payments"]["amount"].tolist() == [500.0]
        assert tables["info"].attrs["cached"]

    def test_cache_reset_when_osv_policy_changes(self, data_dir, settings, tmp_path):
        cache = tmp_path / "cache"
        loaders.load_all(data_dir, settings, cache)
        settings["osv"]["duplicate_month"] = "drop"
        tables = loaders.load_all(data_dir, settings, cache)
        assert not tables["osv"].attrs["cached"] and tables["osv"]["month"].nunique() == 2

    def test_cache_reset_when_loader_version_changes(self, data_dir, settings, tmp_path, monkeypatch):
        cache = tmp_path / "cache"
        loaders.load_all(data_dir, settings, cache)
        monkeypatch.setattr(loaders, "LOADER_VERSION", loaders.LOADER_VERSION + 1)
        assert not loaders.load_all(data_dir, settings, cache)["info"].attrs["cached"]

    def test_broken_cache_is_ignored(self, data_dir, settings, tmp_path):
        cache = tmp_path / "cache"
        loaders.load_all(data_dir, settings, cache)
        (cache / "info.json").write_text("не json", encoding="utf-8")
        tables = loaders.load_all(data_dir, settings, cache)
        assert not tables["info"].attrs["cached"]


# ---------------------------------------------------------------------------
# Командная строка
# ---------------------------------------------------------------------------

class TestCommandLine:
    def test_summary_lists_every_table(self, data_dir, settings):
        summary = loaders.summarize(loaders.load_all(data_dir, settings))
        assert summary["таблица"].tolist() == ["info", "osv", "payments", "events", "limits"]
        assert summary.set_index("таблица").loc["osv", "строк"] == 9

    def test_main_ok(self, data_dir, capsys, tmp_path):
        assert loaders.main(["--data", str(data_dir), "--no-cache"]) == 0
        out = capsys.readouterr().out
        assert "payments" in out and "court_result" in out

    def test_main_reports_error_with_code_1(self, tmp_path, caplog):
        with caplog.at_level(logging.ERROR):
            assert loaders.main(["--data", str(tmp_path / "нет"), "--no-cache"]) == 1
        assert "Файлы не прочитаны" in caplog.text


# ---------------------------------------------------------------------------
# Настоящие данные
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tables(real_data_dir):
    return loaders.load_all(real_data_dir, loaders.load_settings(), cache_dir=None)


@pytest.mark.real_data
class TestRealData:

    def test_row_counts(self, tables):
        assert len(tables["payments"]) == 1_565_019
        assert tables["info"]["ls"].is_unique and len(tables["info"]) == 150_787
        assert tables["osv"]["ls"].nunique() == 150_787

    def test_osv_covers_2025_01_to_2026_03(self, tables):
        months = tables["osv"]["month"].drop_duplicates().sort_values()
        assert months.tolist() == list(pd.date_range("2025-01-01", "2026-03-01", freq="MS"))

    def test_events_per_measure(self, tables):
        counts = tables["events"]["measure"].value_counts()
        assert counts.index.isin(loaders.MEASURE_FILES).all() and (counts > 0).all()
        assert counts["sms"] == 30_419 and counts["restriction"] == 2_484

    def test_limits_match_fallback_yaml(self, tables):
        limits = tables["limits"].set_index("measure")["limit"].astype(object).where(lambda s: s.notna(), None).to_dict()
        reserve = yaml.safe_load((loaders.REPO_ROOT / "configs" / "limits.yaml").read_text(encoding="utf-8"))["limits"]
        assert limits == reserve
