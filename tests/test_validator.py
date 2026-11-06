"""Тесты валидатора входных файлов (DATA-2): испорченный файл → понятная ошибка с именем файла и колонки."""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

import pandas as pd
import pytest

from tns import pipeline
from tns.data import loaders, validator
from tns.data.validator import ERROR, WARNING, Problem, ValidationError
from tests.conftest import INFO_HEADER, write_info, write_limits, write_measure, write_payments, write_xlsx

SMS = loaders.MEASURE_FILES["sms"]


def errors_of(data_dir, settings) -> list[str]:
    """Сообщения об ошибках, которые даёт load_all на этой папке."""
    with pytest.raises(ValidationError) as raised:
        loaders.load_all(data_dir, settings)
    return [p.message for p in raised.value.errors]


def the_error(data_dir, settings) -> str:
    found = errors_of(data_dir, settings)
    assert len(found) == 1, found
    return found[0]


# ---------------------------------------------------------------------------
# Критерий готовности: «в файле 06 нет колонки Дата»
# ---------------------------------------------------------------------------

class TestAcceptance:
    def test_file_06_without_date_column(self, data_dir, settings):
        write_xlsx(data_dir / SMS, [["СМС", None], ["ЛС", "Когда"], [10, datetime(2025, 1, 1)]])
        message = the_error(data_dir, settings)
        assert message.startswith("В файле «06 СМС ХК.xlsx» нет колонки «Дата».")
        assert "В шапке найдено: «ЛС», «Когда»." in message

    def test_report_text(self, data_dir, settings):
        write_xlsx(data_dir / SMS, [["СМС", None], ["ЛС", "Когда"], [10, datetime(2025, 1, 1)]])
        with pytest.raises(ValidationError) as raised:
            loaders.load_all(data_dir, settings)
        text = str(raised.value)
        assert text.startswith("Входные файлы не прошли проверку: 1 ошибка.\n  1. В файле «06 СМС ХК.xlsx» нет колонки «Дата».")
        assert text.endswith(f"Исправьте файлы в папке «{data_dir}» и запустите снова.")

    def test_all_broken_files_reported_at_once(self, data_dir, settings):
        write_xlsx(data_dir / SMS, [["СМС", None], ["ЛС", "Когда"], [10, datetime(2025, 1, 1)]])
        (data_dir / loaders.MEASURE_FILES["visit"]).unlink()
        (data_dir / loaders.PAYMENTS_FILE).write_text("", encoding="utf-8")
        write_limits(data_dir / loaders.LIMITS_FILE, [["04 Автодозвон ХК", "много"]])
        with pytest.raises(ValidationError) as raised:
            loaders.load_all(data_dir, settings)
        files = [p.file for p in raised.value.errors]
        assert set(files) == {SMS, loaders.MEASURE_FILES["visit"], loaders.PAYMENTS_FILE, loaders.LIMITS_FILE}
        assert str(raised.value).splitlines()[0] == f"Входные файлы не прошли проверку: {len(files)} ошибок."

    def test_run_py_shows_the_error_and_exits_1(self, data_dir, tmp_path, settings, caplog):
        write_xlsx(data_dir / SMS, [["СМС", None], ["ЛС", "Когда"], [10, datetime(2025, 1, 1)]])
        config = tmp_path / "settings.yaml"
        import yaml
        settings["paths"]["cache_dir"] = str(tmp_path / "cache")
        config.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")
        with caplog.at_level(logging.ERROR):
            code = pipeline.main(["--month", "2025-02", "--data", str(data_dir), "--out", str(tmp_path / "out"),
                                  "--config", str(config)])
        assert code == 1
        assert "В файле «06 СМС ХК.xlsx» нет колонки «Дата»." in caplog.text
        assert "Traceback" not in caplog.text


# ---------------------------------------------------------------------------
# Файл не открывается
# ---------------------------------------------------------------------------

class TestCannotOpen:
    def test_text_instead_of_excel(self, data_dir, settings):
        (data_dir / SMS).write_text("это не таблица", encoding="utf-8")
        message = the_error(data_dir, settings)
        assert "Файл «06 СМС ХК.xlsx» не открывается как книга Excel" in message
        assert "сохраните как «Книга Excel (*.xlsx)»" in message

    def test_truncated_xlsx(self, data_dir, settings):
        content = (data_dir / loaders.INFO_FILE).read_bytes()
        (data_dir / loaders.INFO_FILE).write_bytes(content[: len(content) // 2])
        assert "Файл «01 Общая информация о ЛС ХК.xlsx» не открывается как книга Excel" in the_error(data_dir, settings)

    @pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="права доступа так не проверить")
    def test_no_access(self, data_dir, settings):
        path = data_dir / SMS
        path.chmod(0)
        try:
            assert "нет доступа. Если он открыт в Excel, закройте его" in the_error(data_dir, settings)
        finally:
            path.chmod(0o644)

    def test_missing_file(self, data_dir, settings):
        (data_dir / SMS).unlink()
        assert the_error(data_dir, settings) == (
            f"Не найден входной файл «{SMS}» в папке «{data_dir}». "
            "Положите файл в эту папку с этим именем или укажите другую папку в --data."
        )

    def test_empty_csv(self, data_dir, settings):
        (data_dir / loaders.PAYMENTS_FILE).write_text("", encoding="utf-8")
        assert the_error(data_dir, settings).startswith("Файл «03 Оплаты ХК.csv» пуст.")

    def test_csv_not_utf8(self, data_dir, settings):
        (data_dir / loaders.PAYMENTS_FILE).write_bytes(
            "Номер;Дата оплаты;Сумма;Способ оплаты\n10;15.01.2025;100;5\n".encode("cp1251")
        )
        assert "не читается в кодировке UTF-8" in the_error(data_dir, settings)

    def test_csv_wrong_separator(self, data_dir, settings):
        (data_dir / loaders.PAYMENTS_FILE).write_text(
            "Номер,Дата оплаты,Сумма,Способ оплаты\n10,15.01.2025,100,5\n", encoding="utf-8"
        )
        message = the_error(data_dir, settings)
        assert "нет колонок «Номер», «Дата оплаты», «Сумма», «Способ оплаты»" in message
        assert "разделитель «;»" in message


# ---------------------------------------------------------------------------
# Пустые файлы и шапки
# ---------------------------------------------------------------------------

class TestStructure:
    def test_several_missing_columns_listed(self, data_dir, settings):
        header = [c for c in INFO_HEADER if c not in {"Наличие льгот", "МКД"}]
        write_xlsx(data_dir / loaders.INFO_FILE, [header, ["Нет"] * len(header)])
        assert "В файле «01 Общая информация о ЛС ХК.xlsx» нет колонок «Наличие льгот», «МКД»." in the_error(data_dir, settings)

    def test_info_only_header(self, data_dir, settings):
        write_info(data_dir / loaders.INFO_FILE, [])
        assert the_error(data_dir, settings) == (
            "В файле «01 Общая информация о ЛС ХК.xlsx» нет ни одной строки с данными, только шапка."
            " Выгрузите файл заново: похоже, выгрузка прервалась или сделана с пустым фильтром."
        )

    def test_payments_only_header(self, data_dir, settings):
        write_payments(data_dir / loaders.PAYMENTS_FILE, [])
        assert "нет ни одной строки с оплатами" in the_error(data_dir, settings)

    def test_osv_without_header(self, data_dir, settings):
        write_xlsx(data_dir / loaders.OSV_FILE, [["ЛС", "Что-то"], [None, "x"], [1, 2]])
        assert "«Что-то» в первой строке шапки не похоже на месяц" in the_error(data_dir, settings)

    def test_osv_single_row(self, data_dir, settings):
        write_xlsx(data_dir / loaders.OSV_FILE, [["ЛС"]])
        assert "нет двухстрочной шапки" in the_error(data_dir, settings)

    def test_measure_file_empty_sheet(self, data_dir, settings):
        write_xlsx(data_dir / SMS, [])
        assert the_error(data_dir, settings) == (
            "Файл «06 СМС ХК.xlsx» пуст: нужна шапка «ЛС | Дата» и строки с мерами."
            " Выгрузите файл заново: похоже, выгрузка прервалась или сделана с пустым фильтром."
        )

    def test_measure_file_without_rows_is_a_warning(self, data_dir, settings):
        write_measure(data_dir / SMS, "СМС", [])
        tables = loaders.load_all(data_dir, settings)
        assert (tables["events"]["measure"] == "sms").sum() == 0
        assert any(p.file == SMS and "нет ни одной строки с мерами" in p.message for p in tables["events"].attrs["problems"])

    def test_measure_file_without_title_row(self, data_dir, settings):
        write_xlsx(data_dir / SMS, [["ЛС", "Дата"], [10, datetime(2025, 1, 1)], [11, datetime(2025, 2, 1)]])
        events = loaders.load_all(data_dir, settings)["events"]
        assert events.loc[events["measure"] == "sms", "ls"].tolist() == [10, 11]

    def test_limits_errors_listed_together(self, data_dir, settings):
        rows = [["04 Автодозвон ХК", "много"], ["99 Новая", 5], ["06 СМС ХК", None]]
        write_limits(data_dir / loaders.LIMITS_FILE, rows)
        found = errors_of(data_dir, settings)
        assert any("строка 2: для меры «04 Автодозвон ХК» лимит «много»" in m for m in found)
        assert any("строка 3: неизвестная мера «99 Новая»" in m for m in found)
        assert any("строка 4: для меры «06 СМС ХК» лимит «пусто»" in m for m in found)
        assert any("нет лимитов для мер" in m for m in found)

    def test_missing_limits_file_is_a_warning(self, data_dir, settings):
        (data_dir / loaders.LIMITS_FILE).unlink()
        limits = loaders.load_all(data_dir, settings)["limits"]
        assert [p.level for p in limits.attrs["problems"]] == [WARNING]


# ---------------------------------------------------------------------------
# Значения, которые не прочитались
# ---------------------------------------------------------------------------

class TestValues:
    def test_some_bad_dates_are_a_warning_with_examples(self, data_dir, settings):
        rows = [[10 + i, datetime(2025, 1, 1)] for i in range(9)] + [[99, "вчера"]]
        write_measure(data_dir / SMS, "СМС", rows)
        events = loaders.load_all(data_dir, settings)["events"]
        problem = next(p for p in events.attrs["problems"] if p.file == SMS)
        assert problem.level == WARNING and (problem.count, problem.total) == (1, 10)
        assert "В файле «06 СМС ХК.xlsx» в колонке «Дата» не прочитаны 1 из 10 значений (10,0 %)" in problem.message
        assert "строка 12 — «вчера»" in problem.message

    def test_all_bad_dates_are_an_error(self, data_dir, settings):
        write_measure(data_dir / SMS, "СМС", [[10, "вчера"], [11, "завтра"]])
        assert the_error(data_dir, settings) == (
            "В файле «06 СМС ХК.xlsx» в колонке «Дата» не прочитано ни одного значения из 2: ожидается дата. "
            "Проверьте, что в колонке нужные данные и колонки не сдвинуты."
        )

    def test_share_above_threshold_is_an_error(self, data_dir, settings):
        write_measure(data_dir / SMS, "СМС", [[10, datetime(2025, 1, 1)], [11, "x"], [12, "y"]])
        message = the_error(data_dir, settings)
        assert "не прочитаны 2 из 3 значений (66,7 %)" in message
        assert "больше допустимого (50,0 %, настройка validation.max_lost_share)" in message

    def test_threshold_is_configurable(self, data_dir, settings):
        write_measure(data_dir / SMS, "СМС", [[10, datetime(2025, 1, 1)], [11, "x"], [12, "y"]])
        settings["validation"]["max_lost_share"] = 0.9
        loaders.load_all(data_dir, settings)

    def test_no_ls_in_any_row(self, data_dir, settings):
        write_info(data_dir / loaders.INFO_FILE, [["abc", *["Нет"] * 16, "g"]])
        assert "в колонке «ЛС» не прочитано ни одного значения из 1: ожидается номер ЛС" in the_error(data_dir, settings)

    def test_flags_column_with_garbage(self, data_dir, settings):
        rows = [[ls, "1", *["Нет"] * 15, f"g{ls}"] for ls in (1, 2)]
        write_xlsx(data_dir / loaders.INFO_FILE, [INFO_HEADER, *rows])
        message = the_error(data_dir, settings)
        assert "«Возможность дистанционного отключения»" in message and "ожидается «Да» или «Нет»" in message

    def test_bad_payment_method_never_blocks(self, data_dir, settings):
        write_payments(data_dir / loaders.PAYMENTS_FILE, ["10;15.01.2025;100;карта", "20;16.01.2025;200;наличные"])
        payments = loaders.load_all(data_dir, settings)["payments"]
        assert payments["payment_method"].isna().all()

    def test_bad_payment_amounts_error(self, data_dir, settings):
        write_payments(data_dir / loaders.PAYMENTS_FILE, ["10;15.01.2025;сто;5"])
        message = the_error(data_dir, settings)
        assert "«03 Оплаты ХК.csv» в колонке «Сумма»" in message and "ожидается число, например 1240,50" in message

    def test_text_dates_in_measure_files_are_read(self, data_dir, settings):
        write_measure(data_dir / SMS, "СМС", [[10, "15.01.2025"], [11, "2025-02-01"]])
        events = loaders.load_all(data_dir, settings)["events"]
        assert events.loc[events["measure"] == "sms", "event_date"].tolist() == [
            pd.Timestamp("2025-01-15"), pd.Timestamp("2025-02-01"),
        ]


# ---------------------------------------------------------------------------
# Помощники валидатора
# ---------------------------------------------------------------------------

class TestHelpers:
    @pytest.mark.parametrize("n, word", [(1, "ошибка"), (2, "ошибки"), (5, "ошибок"), (11, "ошибок"), (21, "ошибка"), (104, "ошибки")])
    def test_plural(self, n, word):
        assert validator.plural(n, "ошибка", "ошибки", "ошибок") == word

    def test_percent(self):
        assert validator.percent(0.41) == "41,0 %" and validator.percent(0.00002) == "менее 0,1 %"

    def test_problem_survives_cache_json(self):
        problem = Problem(WARNING, "06 СМС ХК.xlsx", "текст", column="Дата", count=1, total=10, expected="дата")
        assert Problem.from_dict(problem.to_dict()) == problem

    def test_escalate_keeps_small_losses_as_warnings(self):
        small = Problem(WARNING, "f", "m", "c", 1, 10)
        assert validator.escalate([small], 0.5) == [small]

    def test_report_counts_warnings(self):
        text = validator.format_report([validator.error("f", "Ошибка."), validator.warning("f", "Внимание.")])
        assert "1 ошибка" in text and "Кроме того, 1 предупреждение" in text
        assert [p.level for p in [validator.error("f", "x"), validator.warning("f", "y")]] == [ERROR, WARNING]
