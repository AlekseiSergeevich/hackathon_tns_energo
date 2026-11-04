"""Тесты загрузчика таблицы правил (RULE-1)."""

from pathlib import Path

import openpyxl
import pytest

from tns.rules.loader import RuleBookError, load_rulebook, main

DOCS = Path(__file__).resolve().parents[1] / "docs"
BOOK_V03 = DOCS / "Правила воздействия ТНС v0.3.xlsx"
BOOK_V02 = DOCS / "Правила воздействия ТНС v0.2.xlsx"


@pytest.fixture(scope="module")
def book():
    return load_rulebook(BOOK_V03)


# ---------------------------------------------------------------------------
# Помощники для испорченных копий
# ---------------------------------------------------------------------------


def _col(ws, header):
    for c in range(1, ws.max_column + 1):
        if ws.cell(1, c).value == header:
            return c
    raise KeyError(header)


def _row(ws, code):
    for r in range(2, ws.max_row + 1):
        if ws.cell(r, 1).value == code:
            return r
    raise KeyError(code)


def broken_copy(tmp_path, change):
    """Копия v0.3, испорченная функцией change(wb)."""
    wb = openpyxl.load_workbook(BOOK_V03)
    change(wb)
    path = tmp_path / "Правила воздействия ТНС v9.9.xlsx"
    wb.save(path)
    return path


def errors_of(path):
    with pytest.raises(RuleBookError) as exc:
        load_rulebook(path)
    return exc.value


# ---------------------------------------------------------------------------
# Корректная книга
# ---------------------------------------------------------------------------


def test_v03_counts(book):
    assert len(book.params) == 17
    assert list(book.stages) == ["S0", "S1", "S2", "S3", "S4", "S5", "SX"]
    assert len(book.transitions) == 10
    assert len(book.exceptions) == 8
    assert len(book.rules) == 20
    assert len(book.active_rules) == 18
    assert {c for c, r in book.rules.items() if not r.is_active} == {"R-RCP-01", "R-CRR-01"}
    assert len(book.limits) == 10
    assert len(book.examples) == 9


def test_v03_has_no_warnings(book):
    assert book.warnings == []


def test_version(book):
    assert book.version.label == "0.3"
    assert len(book.version.sha256) == 64
    assert "v0.3" in str(book.version)


def test_sms_rule(book):
    r = book.rules["R-SMS-01"]
    assert r.measure == "sms"
    assert r.stages == ("S1", "S2", "S3", "S4")
    assert (r.overdue_from, r.overdue_to) == (1, 2)
    assert (r.dz_from, r.dz_to) == (500, 2000)
    assert r.condition == "has_phone"
    assert r.is_background is False and r.is_active is True
    assert "overdue_m" in r.placeholders and "dz" in r.placeholders


def test_empty_thresholds_are_none(book):
    r = book.rules["R-EML-01"]
    assert r.overdue_to is None and r.dz_to is None
    assert r.is_background is True


def test_same_as_above_is_resolved(book):
    first = book.rules["R-CRT-01"].condition
    assert first and "то же" not in first.lower()
    assert book.rules["R-CRT-02"].condition == first
    assert book.rules["R-CRT-03"].condition == first
    assert book.rules["R-CRT-04"].measure == "commission_list"


def test_ntc_uses_single_norm(book):
    assert "P-08 * norm" in book.rules["R-NTC-01"].condition


def test_params(book):
    assert book.params["P-01"].value_num == 500
    assert book.params["P-11"].value_num == 30000
    assert book.params["P-02"].value_num is None  # текстовый параметр
    assert "FIFO" in book.params["P-02"].value_raw


def test_limits(book):
    assert book.limits["email"].value is None
    assert book.limits["restriction"].value == 200
    assert book.limits["sms"].value == 2150
    assert book.limits["court_application"].rule_codes == ("R-CRT-01", "R-CRT-02", "R-CRT-03")
    assert book.limits["visit"].rule_codes == tuple(f"R-VST-0{i}" for i in range(1, 6))
    assert "commission_list" not in book.limits  # не расходует лимит заявлений


def test_transitions(book):
    by_code = {t.code: t for t in book.transitions}
    assert by_code["T-0→1"].from_stages == ("S0",) and by_code["T-0→1"].to_stage == "S1"
    assert by_code["T-*→0"].from_stages == tuple(book.stages)
    assert by_code["T-*→1"].from_stages == ("S2", "S3")
    back = by_code["T-X→пред."]
    assert back.returns_to_previous and back.to_stage is None
    assert [t.order for t in book.transitions] == sorted(t.order for t in book.transitions)


def test_exceptions(book):
    assert book.exceptions["EX-06"].fields == ("court_app_date", "court_result_date")
    assert "EX-08" in book.exceptions


def test_fields(book):
    names = book.field_names
    for name in ("dz", "norm", "has_phone", "has_ebill", "pay30_m", "call_failed", "refused", "*_date"):
        assert name in names
    for name in ("has_mobile", "has_email", "norm2x"):
        assert name not in names


def test_v02_still_loads_with_warnings():
    old = load_rulebook(BOOK_V02)
    assert len(old.rules) == 20
    assert any("court_threshold_ok" in w for w in old.warnings)


# ---------------------------------------------------------------------------
# Испорченные копии
# ---------------------------------------------------------------------------


def test_missing_column(tmp_path):
    def change(wb):
        ws = wb["Правила"]
        ws.delete_cols(_col(ws, "ДЗ от, руб."))

    err = errors_of(broken_copy(tmp_path, change))
    assert any("«Правила»" in e and "«ДЗ от, руб.»" in e for e in err.errors)


def test_missing_sheet(tmp_path):
    err = errors_of(broken_copy(tmp_path, lambda wb: wb.remove(wb["Лимиты"])))
    assert "нет листа «Лимиты»" in err.errors


def test_duplicate_code(tmp_path):
    def change(wb):
        ws = wb["Правила"]
        ws.cell(_row(ws, "R-ADL-01"), 1).value = "R-SMS-01"

    err = errors_of(broken_copy(tmp_path, change))
    assert any("R-SMS-01 уже встречался" in e for e in err.errors)


def test_bad_code_pattern(tmp_path):
    def change(wb):
        ws = wb["Правила"]
        ws.cell(_row(ws, "R-SMS-01"), 1).value = "SMS-1"

    err = errors_of(broken_copy(tmp_path, change))
    assert any("«SMS-1» не по шаблону" in e for e in err.errors)


def test_text_in_threshold(tmp_path):
    def change(wb):
        ws = wb["Правила"]
        ws.cell(_row(ws, "R-SMS-01"), _col(ws, "ДЗ от, руб.")).value = "пятьсот"

    err = errors_of(broken_copy(tmp_path, change))
    assert any("«ДЗ от, руб.»" in e and "«пятьсот»" in e for e in err.errors)


def test_from_greater_than_to(tmp_path):
    def change(wb):
        ws = wb["Правила"]
        ws.cell(_row(ws, "R-SMS-01"), _col(ws, "ДЗ от, руб.")).value = 5000

    err = errors_of(broken_copy(tmp_path, change))
    assert any("«от» (5000) больше, чем «до» (2000)" in e for e in err.errors)


def test_unknown_stage(tmp_path):
    def change(wb):
        ws = wb["Правила"]
        ws.cell(_row(ws, "R-SMS-01"), _col(ws, "Стадии")).value = "S1–S7"

    err = errors_of(broken_copy(tmp_path, change))
    assert any("«S7» не найдена" in e for e in err.errors)


def test_unknown_param_reference(tmp_path):
    def change(wb):
        ws = wb["Правила"]
        ws.cell(_row(ws, "R-NTC-01"), _col(ws, "Доп. условие (формально)")).value = "dz > P-99"

    err = errors_of(broken_copy(tmp_path, change))
    assert any("R-NTC-01 ссылается на параметр P-99" in e for e in err.errors)


def test_bad_yes_no(tmp_path):
    def change(wb):
        ws = wb["Правила"]
        ws.cell(_row(ws, "R-SMS-01"), _col(ws, "Активно")).value = "может быть"

    err = errors_of(broken_copy(tmp_path, change))
    assert any("«Да» или «Нет»" in e and "«может быть»" in e for e in err.errors)


def test_unknown_measure(tmp_path):
    def change(wb):
        ws = wb["Правила"]
        ws.cell(_row(ws, "R-SMS-01"), _col(ws, "Мера")).value = "Голубиная почта"

    err = errors_of(broken_copy(tmp_path, change))
    assert any("неизвестная мера «Голубиная почта»" in e for e in err.errors)


def test_bad_limit(tmp_path):
    def change(wb):
        ws = wb["Лимиты"]
        ws.cell(_row(ws, "06 СМС"), _col(ws, "Лимит в месяц")).value = "много"

    err = errors_of(broken_copy(tmp_path, change))
    assert any("«Лимиты»" in e and "«много»" in e for e in err.errors)


def test_limit_mask_matches_nothing(tmp_path):
    def change(wb):
        ws = wb["Лимиты"]
        ws.cell(_row(ws, "06 СМС"), _col(ws, "Код правил")).value = "R-XYZ-*"

    err = errors_of(broken_copy(tmp_path, change))
    assert any("«R-XYZ-*» не соответствует ни одному правилу" in e for e in err.errors)


def test_all_errors_reported_at_once(tmp_path):
    def change(wb):
        ws = wb["Правила"]
        ws.cell(_row(ws, "R-SMS-01"), _col(ws, "ДЗ от, руб.")).value = "пятьсот"
        ws.cell(_row(ws, "R-ADL-01"), _col(ws, "Активно")).value = "?"

    err = errors_of(broken_copy(tmp_path, change))
    assert len(err.errors) == 2
    assert "ошибки (2)" in str(err)


def test_unknown_field_is_warning_not_error(tmp_path):
    def change(wb):
        ws = wb["Правила"]
        ws.cell(_row(ws, "R-SMS-01"), _col(ws, "Доп. условие (формально)")).value = "has_mobile"

    book = load_rulebook(broken_copy(tmp_path, change))
    assert any("has_mobile" in w for w in book.warnings)


def test_footnote_rows_are_skipped(tmp_path):
    def change(wb):
        ws = wb["Правила"]
        ws.cell(ws.max_row + 2, 2).value = "Важно: это сноска, а не правило"

    assert len(load_rulebook(broken_copy(tmp_path, change)).rules) == 20


def test_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rulebook(tmp_path / "нет такого.xlsx")


def test_not_an_excel_file(tmp_path):
    path = tmp_path / "правила.xlsx"
    path.write_text("это не Excel", encoding="utf-8")
    err = errors_of(path)
    assert "не удалось открыть как Excel" in err.errors[0]


# ---------------------------------------------------------------------------
# Консоль
# ---------------------------------------------------------------------------


def test_cli_ok(capsys):
    assert main([str(BOOK_V03)]) == 0
    out = capsys.readouterr().out
    assert "Правила:     20  (активных 18" in out
    assert "Предупреждений нет." in out


def test_cli_error(tmp_path, capsys):
    path = broken_copy(tmp_path, lambda wb: wb.remove(wb["Лимиты"]))
    assert main([str(path)]) == 1
    assert "нет листа «Лимиты»" in capsys.readouterr().err
