"""Срок просрочки по P-02, FIFO (DATA-5): ручной расчёт на 10 ЛС и сверка с панелью.

Месяцы с 2025-01 (январь — t=0). Строка ОСВ — (сальдо на 01-е, начислено, оплачено).
На 01.M счёт за M−1 ещё не просрочен (оплата до 15.M). Срок — от самого старого
непогашенного счёта до M−2 включительно. Оплаты гасят самые старые счета.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from tns.data import loaders, overdue_check, panel, quality
from tests.conftest import make_osv
from tests.test_panel import build, column

N = None  # срок не определить

# (ситуация, строки ОСВ по месяцам, срок, просроченная сумма, долг старше данных)
CASES = [
    (
        # Не платит. 01.03: долг 200, счёт февраля не просрочен, январь — 1 мес. Дальше +1 каждый месяц.
        "1. долг растёт без оплат",
        [(0, 100, 0), (100, 100, 0), (200, 100, 0), (300, 100, 0), (400, 100, 0), (500, 100, 0)],
        [0, 0, 1, 2, 3, 4], [0, 0, 100, 200, 300, 400], [False] * 6,
    ),
    (
        # Платит по 50 с февраля. 01.03: январь 50 (1 мес.). 01.04: январь погашен, февраль 100 (1 мес.).
        # 01.05: февраль 50 + март 100 (2 мес.). 01.06: март 100 + апрель 100 (2 мес.).
        "2. частичные оплаты гасят самые старые счета",
        [(0, 100, 0), (100, 100, 50), (150, 100, 50), (200, 100, 50), (250, 100, 50), (300, 100, 50)],
        [0, 0, 1, 1, 2, 2], [0, 0, 50, 100, 150, 200], [False] * 6,
    ),
    (
        # Январь 300 не оплачен, дальше платит ровно по 100. По FIFO эти 100 идут в январь,
        # поэтому срок не растёт бесконечно: 01.04 — январь 100 + февраль 100 (2 мес.), 01.05 — февраль+март (2 мес.).
        "3. платит «текущий» счёт, оплата уходит в старый",
        [(0, 300, 0), (300, 100, 100), (300, 100, 100), (300, 100, 100), (300, 100, 100), (300, 100, 100)],
        [0, 0, 1, 2, 2, 2], [0, 0, 200, 200, 200, 200], [False] * 6,
    ),
    (
        # В марте оплатил 300: всё погашено. 01.06: апрель 100 просрочен (1 мес.), май ещё нет.
        "4. погасил долг, потом новый",
        [(0, 100, 0), (100, 100, 0), (200, 100, 300), (0, 100, 0), (100, 100, 0), (200, 100, 0)],
        [0, 0, 1, 0, 0, 1], [0, 0, 100, 0, 0, 100], [False] * 6,
    ),
    (
        # Переплата 200 в январе гасит счета февраля и марта. Апрель не оплачен: 01.06 — 1 мес.
        "5. переплата гасит следующие счета",
        [(0, 100, 300), (-200, 100, 0), (-100, 100, 0), (0, 100, 0), (100, 100, 0), (200, 100, 0)],
        [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 100], [False] * 6,
    ),
    (
        # В феврале оплачены январь и февраль. В апреле возврат 200: снова открываются
        # последние оплаченные счета — февраль и январь. 01.05: январь…март (3 мес.), 01.06 — 4 мес.
        "6. возврат оплаты открывает счета заново",
        [(0, 100, 0), (100, 100, 200), (0, 100, 0), (100, 100, -200), (400, 100, 0), (500, 100, 0)],
        [0, 0, 0, 0, 3, 4], [0, 0, 0, 0, 300, 400], [False] * 6,
    ),
    (
        # Февраль и апрель без начислений. Режим count: срок — месяцы с даты долга (январь), пустые считаются.
        "7. месяцы без начисления (count)",
        [(0, 100, 0), (100, 0, 0), (100, 100, 0), (200, 0, 0), (200, 100, 0), (300, 0, 0)],
        [0, 0, 1, 2, 3, 4], [0, 0, 100, 100, 200, 200], [False] * 6,
    ),
    (
        # В марте перерасчёт −150 гасит январь (100) и половину февраля. 01.04: февраль 50 (1 мес.).
        # 01.05: февраль 50 (2 мес.: февраль и пустой март). 01.06: февраль 50 + апрель 100 (3 мес.).
        "8. отрицательный перерасчёт гасит старые счета",
        [(0, 100, 0), (100, 100, 0), (200, -150, 0), (50, 100, 0), (150, 100, 0), (250, 100, 0)],
        [0, 0, 1, 1, 2, 3], [0, 0, 100, 50, 50, 150], [False] * 6,
    ),
    (
        # Долг 1 000 был до начала данных. Каждый месяц платит 100 — уходит в этот старый долг.
        # Январь: начисление за декабрь неизвестно, срок не определить. Дальше срок = месяцев с начала данных.
        "9. долг старше начала данных",
        [(1000, 100, 100)] * 6,
        [N, 1, 2, 3, 4, 5], [N, 900, 900, 900, 900, 900], [True] * 6,
    ),
    (
        # Всё оплачено, но в апреле сальдо 500 без начисления (корректировка +500). Она открывает
        # заново март, февраль, январь (300), остальные 200 — долг до начала данных.
        # 01.04: 3 мес., долг старше данных. Оплаты по 100 гасят его, к 01.06 самый старый — январь (4 мес.).
        "10. сальдо выросло без начисления (корректировка)",
        [(0, 100, 100), (0, 100, 100), (0, 100, 100), (500, 100, 100), (500, 100, 100), (500, 100, 100)],
        [0, 0, 0, 3, 4, 4], [0, 0, 0, 400, 400, 400], [False, False, False, True, True, False],
    ),
]


@pytest.mark.parametrize("name, rows, months, overdue, censored", CASES, ids=[c[0] for c in CASES])
def test_manual_case(name, rows, months, overdue, censored):
    tables = build(make_osv({1: rows}))
    assert column(tables, 1, "overdue_m") == months
    assert column(tables, 1, "dz_overdue") == overdue
    assert column(tables, 1, "overdue_censored") == censored
    # Тот же ответ даёт пошаговый расчёт по очереди счетов.
    saldo, accrual, paid = zip(*rows)
    states = overdue_check.fifo_by_bills(saldo, accrual, paid)
    assert [None if np.isnan(s.overdue_m) else s.overdue_m for s in states] == months
    assert [None if np.isnan(s.dz_overdue) else s.dz_overdue for s in states] == overdue
    assert [s.censored for s in states] == censored


def test_zero_accrual_months_skip(settings):
    # Пример 7 в режиме skip: пустые месяцы не считаются. 01.05: январь и март — 2 мес.
    rows = CASES[6][1]
    settings["overdue"]["zero_accrual_months"] = "skip"
    assert column(build(make_osv({1: rows}), settings=settings), 1, "overdue_m") == [0, 0, 1, 1, 2, 2]
    states = overdue_check.fifo_by_bills(*zip(*rows), zero_accrual_months="skip")
    assert [s.overdue_m for s in states] == [0, 0, 1, 1, 2, 2]


def test_allocation_is_a_parameter(settings):
    # Правило разнесения оплат — параметр overdue.allocation. Другое значение пока не поддерживается.
    assert settings["overdue"]["allocation"] == "fifo"
    settings["overdue"]["allocation"] = "by_period"
    with pytest.raises(ValueError, match="overdue.allocation.*fifo"):
        build(make_osv({1: CASES[0][1]}), settings=settings)


def random_osv(n_ls: int, n_months: int, seed: int) -> pd.DataFrame:
    """Случайные ЛС со всем подряд: пустые и отрицательные начисления, переплаты, возвраты, корректировки."""
    rng = np.random.default_rng(seed)
    rows = {}
    for ls in range(1, n_ls + 1):
        saldo = float(rng.choice([0, 0, rng.integers(-300, 3000)]))
        months = []
        for _ in range(n_months):
            accrual = float(rng.choice([0, rng.integers(50, 500), rng.integers(50, 500), -rng.integers(10, 300)]))
            paid = float(rng.choice([0, 0, rng.integers(10, 800), rng.integers(10, 800), -rng.integers(10, 500)]))
            months.append((round(saldo, 2), accrual, paid))
            correction = float(rng.choice([0] * 8 + [rng.integers(-300, 300)]))
            saldo = saldo + accrual - paid + correction
        rows[ls] = months
    return make_osv(rows)


@pytest.mark.parametrize("mode", ["count", "skip"])
def test_panel_equals_manual_on_random_accounts(settings, mode):
    settings["overdue"]["zero_accrual_months"] = mode
    osv = random_osv(500, 10, seed=5)
    p = build(osv, settings=settings)["panel"]
    _, _, m = overdue_check._matrices(p)
    for i in range(m["saldo_start"].shape[0]):
        states = overdue_check.fifo_by_bills(m["saldo_start"][i], m["accrual"][i], m["paid"][i], mode)
        np.testing.assert_array_equal([s.overdue_m for s in states], m["overdue_m"][i], err_msg=f"ЛС {i + 1}")
        np.testing.assert_allclose([s.dz_overdue for s in states], m["dz_overdue"][i], atol=0.011, err_msg=f"ЛС {i + 1}")


class TestCheckCli:
    def write_panel(self, tmp_path):
        tables = build(make_osv({ls: rows for ls, (_, rows, *_) in enumerate(CASES, start=1)}))
        folder = tmp_path / "2025-06"
        folder.mkdir()
        panel.write_parquet(tables["panel"], folder / "panel.parquet")
        return folder

    def test_writes_check_for_given_accounts(self, tmp_path):
        folder = self.write_panel(tmp_path)
        assert overdue_check.main(["--month", "2025-06", "--out", str(tmp_path), "--ls", "3", "6"]) == 0
        table = pd.read_excel(folder / "overdue_check.xlsx", sheet_name="Сверка")
        assert table["ЛС"].drop_duplicates().tolist() == [3, 6]
        assert table["Совпало"].all()
        # Пример 6, 01.05: после возврата снова открыты январь, февраль, март и апрель.
        may = table[(table["ЛС"] == 6) & (table["Месяц (на 01-е)"] == "2025-05")].iloc[0]
        assert may["Непогашенные счета на 01-е"] == "2025-01: 100.00; 2025-02: 100.00; 2025-03: 100.00; 2025-04: 100.00"
        assert may["Срок, мес. (ручной)"] == 3

    def test_unknown_account_and_missing_panel(self, tmp_path, caplog):
        self.write_panel(tmp_path)
        with caplog.at_level(logging.ERROR):
            assert overdue_check.main(["--month", "2025-06", "--out", str(tmp_path), "--ls", "999"]) == 1
            assert overdue_check.main(["--month", "2025-07", "--out", str(tmp_path)]) == 1
        assert "В панели нет ЛС: 999" in caplog.text
        assert "Сначала постройте панель" in caplog.text


@pytest.mark.real_data
def test_real_data_manual_check(real_data_dir, settings):
    raw, _ = quality.clean(loaders.load_all(real_data_dir, settings, loaders.default_cache_dir(settings)), settings)
    p = panel.build_from_raw(raw, settings)["panel"]
    cases = overdue_check.pick_cases(p)
    assert len(cases) == 10
    assert overdue_check.check_table(p, cases)["Совпало"].all()
    # И на всех ЛС ручной расчёт совпадает с панелью.
    _, _, m = overdue_check._matrices(p)
    for i in range(m["saldo_start"].shape[0]):
        states = overdue_check.fifo_by_bills(m["saldo_start"][i], m["accrual"][i], m["paid"][i])
        assert np.array_equal(np.array([s.overdue_m for s in states]), m["overdue_m"][i], equal_nan=True)
