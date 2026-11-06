"""Тесты машины стадий (tns/rules/stages.py) на маленьких придуманных ЛС.

У каждого ЛС заранее известен правильный путь по стадиям. Пороги — из таблицы правил v0.3:
P-01 = 500, P-05 = 1 мес., P-06 = 20 дней, P-08 = 2, пороги суда R-CRT-01…03.
"""

from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pytest

from tns.data.loaders import load_settings
from tns.pipeline import RunContext
from tns.rules import stages as S
from tns.rules.loader import load_rulebook

BOOK = Path(__file__).resolve().parents[1] / "docs" / "Правила воздействия ТНС v0.3.xlsx"
MONTHS = pd.date_range("2025-01-01", "2025-10-01", freq="MS")
NA = np.nan

# ЛС → помесячно (январь … октябрь 2025): долг на 01.M, срок просрочки, норматив.
SCENARIOS = {
    # Полный путь: S0 → S1 → S2 → (уведомление) → S3 → (ограничение) → S4 → погашение → новый цикл.
    1: dict(dz=[300, 1800, 2900, 4000, 4500, 5200, 12000, 0, 0, 1200],
            overdue=[NA, 1, 2, 3, 4, 5, 6, 0, 0, 1],
            norm=[NA, NA, 800, 800, 800, 800, 800, 800, 800, 800]),
    # Без откатов: долг упал до 1 000 ₽ (≥ 500), уведомления не было — остаётся в S2.
    2: dict(dz=[5000, 5000, 5000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
            overdue=[NA, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            norm=[NA, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
    # Ограничение в апреле видно только с 01.05 (по факту S3); норматива нет — T-1→2 не срабатывает.
    3: dict(dz=[3000] * 10, overdue=[NA, 1, 2, 3, 4, 5, 6, 7, 8, 9], norm=[NA] * 10),
    # Претензия в феврале, погашение в марте: дата претензии не сбрасывается.
    4: dict(dz=[900, 900, 0, 0, 0, 0, 0, 0, 0, 0], overdue=[NA, 1, 0, 0, 0, 0, 0, 0, 0, 0], norm=[NA] * 10),
}
EVENTS = [
    (1, "sms", "2025-02-01", False),
    (1, "restriction_notice", "2025-03-26", False),
    (1, "restriction_notice", "2025-04-15", False),  # повторное уведомление: в цикле берётся первое
    (1, "restriction", "2025-05-14", False),
    (3, "restriction", "2025-04-10", False),
    (4, "claim", "2025-02-05", False),
]


def make_panel() -> pd.DataFrame:
    rows = []
    for ls, sc in SCENARIOS.items():
        for i, m in enumerate(MONTHS):
            rows.append(dict(ls=ls, month=m, dz=float(sc["dz"][i]), overdue_m=sc["overdue"][i], norm=sc["norm"][i],
                             months_no_payment=NA, dz_overdue=NA, pay30_m=False))
    p = pd.DataFrame(rows)
    p["overdue_m"] = p["overdue_m"].astype("Float64").round().astype("Int64")
    p["pay30_m"] = p["pay30_m"].astype("boolean")
    return p


def make_events() -> pd.DataFrame:
    e = pd.DataFrame(EVENTS, columns=["ls", "measure", "event_date", "result_30pct"])
    e["event_date"] = pd.to_datetime(e["event_date"])
    e["month"] = e["event_date"].dt.to_period("M").dt.to_timestamp()
    e["result_30pct"] = e["result_30pct"].astype("boolean")
    return e


@pytest.fixture(scope="module")
def book():
    return load_rulebook(BOOK)


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def history(book, settings):
    return S.build_history(make_panel(), make_events(), book, settings)


def path(history, ls, column="stage"):
    return [None if pd.isna(v) else str(v) for v in history[history["ls"] == ls].sort_values("month")[column]]


def test_columns_follow_contract(history):
    assert list(history.columns) == S.STAGES_COLUMNS
    assert len(history) == len(SCENARIOS) * len(MONTHS)
    assert str(history["stage"].dtype) == "category"
    assert history["months_in_stage"].dtype == "int64"


def test_full_path(history):
    assert path(history, 1) == ["S0", "S1", "S2", "S2", "S3", "S3", "S4", "S0", "S0", "S1"]
    assert path(history, 1, "transition") == [None, "T-0→1", "T-1→2", None, "T-2→3", None, "T-3→4", "T-*→0", None, "T-0→1"]


def test_cycle_and_dates(history):
    h = history[history["ls"] == 1].set_index("month")
    assert h.loc["2025-02-01", "cycle_id"] == 1 and h.loc["2025-02-01", "cycle_start"] == pd.Timestamp("2025-02-01")
    # первое уведомление цикла, а не повторное
    assert h.loc["2025-05-01", "notice_date"] == pd.Timestamp("2025-03-26")
    assert h.loc["2025-06-01", "restriction_date"] == pd.Timestamp("2025-05-14")
    # погашение: цикл закрыт, даты цикла сброшены
    assert pd.isna(h.loc["2025-08-01", "notice_date"]) and pd.isna(h.loc["2025-08-01", "restriction_date"])
    assert pd.isna(h.loc["2025-08-01", "cycle_start"])
    # новый долг — новый цикл без старых дат
    assert h.loc["2025-10-01", "cycle_id"] == 2 and pd.isna(h.loc["2025-10-01", "notice_date"])


def test_notice_waits_20_days(history):
    h = history[history["ls"] == 1].set_index("month")
    # 01.04: с уведомления 26.03 прошло 6 дней — ещё S2
    assert h.loc["2025-04-01", "stage"] == "S2" and pd.isna(h.loc["2025-04-01", "transition"])


def test_court_thresholds(history):
    h = history[history["ls"] == 1].set_index("month")
    # 01.06: долг 5 200 ₽ при просрочке 5 мес. — порогов суда нет, остаётся S3
    assert h.loc["2025-06-01", "stage"] == "S3"
    # 01.07: долг 12 000 ₽ при просрочке 6 мес. — R-CRT-01 (≥ 10 000 и ≥ 4 мес.)
    assert h.loc["2025-07-01", "transition"] == "T-3→4"


def test_one_step_per_month_and_no_rollback(history):
    # в феврале условия S2 тоже выполнены, но за месяц — только один шаг
    assert path(history, 2)[:3] == ["S0", "S1", "S2"]
    # долг упал до 1 000 ₽: без отката, остаётся S2
    assert set(path(history, 2)[3:]) == {"S2"}
    assert set(path(history, 2, "stage_source")[3:]) == {"carry"}


def test_fact_is_visible_only_next_month(history):
    h = history[history["ls"] == 3].set_index("month")
    assert h.loc["2025-04-01", "stage"] == "S1"  # ограничение 10.04 на 01.04 ещё не известно
    assert h.loc["2025-05-01", "stage"] == "S3"
    assert h.loc["2025-05-01", "stage_fact"] == "S3"
    assert h.loc["2025-05-01", "stage_source"] == "fact"
    assert pd.isna(h.loc["2025-05-01", "transition"])


def test_no_norm_no_notice_stage(history):
    # норматива нет — условие dz > P-08 × norm не выполнено, S1 → S2 не происходит
    assert path(history, 3)[1:4] == ["S1", "S1", "S1"]


def test_last_claim_kept_after_close(history):
    h = history[history["ls"] == 4].set_index("month")
    assert h.loc["2025-03-01", "stage"] == "S0" and h.loc["2025-03-01", "transition"] == "T-*→0"
    assert h.loc["2025-06-01", "last_claim_date"] == pd.Timestamp("2025-02-05")


def test_first_month_is_start(history):
    first = history[history["month"] == MONTHS[0]]
    assert set(first["stage_source"].astype(str)) == {"start"}
    assert first["transition"].isna().all()
    assert first["stage_prev"].isna().all()


def test_months_in_stage(history):
    assert path(history, 2, "months_in_stage")[2:6] == ["1", "2", "3", "4"]


def test_no_future_data_leak(book, settings, history):
    """Стадия на 01.M не меняется, если удалить все данные с месяца M+1."""
    cut = pd.Timestamp("2025-05-01")
    panel, events = make_panel(), make_events()
    short = S.build_history(panel[panel["month"] <= cut], events[events["month"] < cut], book, settings)
    full = history[history["month"] <= cut].sort_values(["ls", "month"], ignore_index=True)
    short = short.sort_values(["ls", "month"], ignore_index=True)
    cols = ["stage", "stage_fact", "transition", "cycle_id", "notice_date", "restriction_date"]
    pd.testing.assert_frame_equal(full[cols], short[cols])


def test_reasons(book, settings, history):
    rows = history[(history["ls"] == 1) & history["month"].isin(MONTHS[[2, 4, 7]])]
    texts = S.explain(rows, make_panel(), book, settings).tolist()
    assert texts[0].startswith("T-1→2:") and "долг 2 900 ₽" in texts[0]
    assert "уведомление 26.03.2025" in texts[1]
    assert texts[2].startswith("T-*→0:")


def test_settings_errors(book, settings):
    bad = {**settings, "stages": {**settings["stages"], "measure_stage": {"sms": "S9"}}}
    with pytest.raises(S.StagesError, match="неизвестная стадия «S9»"):
        S.build_model(book, bad)
    with pytest.raises(S.StagesError, match="откаты"):
        S.build_model(book, {**settings, "stages": {**settings["stages"], "rollbacks": True}})


def _ctx(tmp_path, settings, month="2025-08"):
    return RunContext(month=pd.Timestamp(month), data_dir=tmp_path, out_dir=tmp_path / "out" / month,
                      rules_path=BOOK, settings=settings, cache_dir=tmp_path / "cache",
                      tables={"panel": make_panel(), "measure_events": make_events()})


def test_pipeline_step_cache_and_excel(tmp_path, settings, caplog):
    caplog.set_level("INFO")
    stages = S.build_stages(_ctx(tmp_path, settings))
    assert stages["month"].max() == pd.Timestamp("2025-08-01")
    plan = stages[stages["month"] == pd.Timestamp("2025-08-01")]
    assert plan["reason"].notna().all()
    assert (tmp_path / "cache" / "stages_history.parquet").exists()

    wb = openpyxl.load_workbook(tmp_path / "out" / "2025-08" / "stages.xlsx", read_only=True)
    assert wb.sheetnames == ["Сводка", "Текущая стадия", "Переходы"]
    assert sum(1 for _ in wb["Текущая стадия"].iter_rows()) == len(SCENARIOS) + 1
    transitions = list(wb["Переходы"].iter_rows(values_only=True))
    assert transitions[0] == ("ЛС", "Месяц", "Было", "Стало", "Код", "Объяснение")
    codes = {row[4] for row in transitions[1:]}
    assert "T-1→2" in codes and S.FACT_LABEL in codes  # ЛС 3: S1 → S3 по факту ограничения

    caplog.clear()
    S.build_stages(_ctx(tmp_path, settings))
    assert "из кэша" in caplog.text


def test_cache_invalidated_by_settings(tmp_path, settings, caplog):
    caplog.set_level("INFO")
    S.build_stages(_ctx(tmp_path, settings))
    changed = {**settings, "stages": {**settings["stages"], "excel": False}}
    caplog.clear()
    S.build_stages(_ctx(tmp_path, changed))
    assert "посчитана" in caplog.text


def test_month_outside_data(tmp_path, settings):
    with pytest.raises(S.StagesError, match="нет стадий на 2026-05"):
        S.build_stages(_ctx(tmp_path, settings, "2026-05"))
