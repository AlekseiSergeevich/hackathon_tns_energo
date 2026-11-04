"""Очистка прочитанных файлов и отчёт о качестве данных (DATA-3).

clean() работает с таблицами load_all до сборки панели и возвращает очищенные таблицы и отчёт.
Что убирается:
    Q-09, Q-10  оплаты и меры с датой позже даты запуска (quality.as_of);
    Q-13        ЛС без движения: во всей ОСВ нулевые сальдо, начисления и оплаты, нет оплат и мер
                (если quality.drop_inactive_ls: true).
Повторы ЛС в файлах 01 и 02 убирает загрузчик (оставляет первую строку), они попадают в отчёт
из его предупреждений. Остальные проверки только считаются: отрицательное сальдо по контракту
остаётся в saldo_start, а в долг не входит (dz = max(0, saldo_start), docs/interfaces.md, раздел 2).

Отчёт: таблица quality (одна строка — одна проверка, даже если ничего не найдено) и примеры
quality_examples; write_report() пишет их в Excel quality_report.xlsx в папку месяца.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tns.data import validator
from tns.data.loaders import MEASURE_FILES

logger = logging.getLogger(__name__)

REPORT_FILE = "quality_report.xlsx"
SUMMARY_COLUMNS = ["code", "check", "source", "count", "total", "share", "action", "details"]
EXAMPLE_COLUMNS = ["code", "ls", "when", "value", "comment"]
SUMMARY_HEADERS = {
    "code": "Код", "check": "Проверка", "source": "Источник", "count": "Найдено", "total": "Из",
    "share": "Доля", "action": "Что сделано", "details": "Подробности",
}
EXAMPLE_HEADERS = {"code": "Код", "ls": "ЛС", "when": "Месяц / дата", "value": "Значение", "comment": "Комментарий"}

KEPT = "оставлено, только в отчёте"
DROPPED = "убрано"


@dataclass
class QualityReport:
    summary: pd.DataFrame
    examples: pd.DataFrame
    as_of: pd.Timestamp


@dataclass
class _Collector:
    limit: int
    rows: list[dict] = field(default_factory=list)
    examples: list[pd.DataFrame] = field(default_factory=list)

    def add(self, code: str, check: str, source: str, count: int, total: int | None, action: str,
            details: str = "", examples: pd.DataFrame | None = None) -> None:
        share = count / total if total else np.nan
        self.rows.append({
            "code": code, "check": check, "source": source, "count": int(count),
            "total": None if total is None else int(total), "share": share, "action": action if count else "—",
            "details": details,
        })
        if count and examples is not None and len(examples):
            frame = examples.head(self.limit).reindex(columns=EXAMPLE_COLUMNS[1:])
            self.examples.append(frame.assign(code=code)[EXAMPLE_COLUMNS])

    def report(self, as_of: pd.Timestamp) -> QualityReport:
        summary = pd.DataFrame(self.rows, columns=SUMMARY_COLUMNS).astype({
            "code": "string", "check": "string", "source": "string", "count": "Int64", "total": "Int64",
            "share": "float64", "action": "string", "details": "string",
        })
        examples = (pd.concat(self.examples, ignore_index=True) if self.examples
                    else pd.DataFrame(columns=EXAMPLE_COLUMNS))
        examples = examples.astype({
            "code": "string", "ls": "Int64", "when": "string", "value": "string", "comment": "string",
        })
        return QualityReport(summary, examples, as_of)


def _month(dates: pd.Series) -> pd.Series:
    return dates.dt.strftime("%Y-%m")


def _money(values: pd.Series) -> pd.Series:
    return values.map(lambda v: f"{v:,.2f}".replace(",", " ").replace(".", ",")).astype("string")


def _ls_examples(ls_values, comment: str = "") -> pd.DataFrame:
    return pd.DataFrame({"ls": pd.Series(np.sort(np.asarray(ls_values)), dtype="Int64"), "comment": comment})


def _with_attrs(table: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    table.attrs = dict(source.attrs)
    return table


def clean(raw: dict[str, pd.DataFrame], settings: dict, as_of: pd.Timestamp | str | None = None
          ) -> tuple[dict[str, pd.DataFrame], QualityReport]:
    """Очищает таблицы load_all и считает отчёт о качестве. Входной словарь не меняется.

    as_of — дата, позже которой даты считаются будущими; по умолчанию quality.as_of или сегодня.
    """
    options = settings["quality"]
    as_of = pd.Timestamp(as_of or options.get("as_of") or pd.Timestamp.today()).normalize()
    found = _Collector(limit=int(options["examples"]))
    info, osv, payments, events = raw["info"], raw["osv"], raw["payments"], raw["events"]

    # Q-00. Предупреждения загрузчиков. Повторы ЛС в 01 и 02 загрузчик уже убрал, оставив первую строку.
    for name, table in raw.items():
        for problem in table.attrs.get("problems", []):
            if problem.expected:
                check, action = f"Не прочитаны значения в колонке «{problem.column}»", "см. подробности"
            elif problem.column == "ЛС" and problem.count:
                check, action = "Повторы ЛС", f"{DROPPED}: оставлена первая строка"
            else:
                check, action = "Замечание при чтении файла", "см. подробности"
            source = problem.file[:2] if problem.file[:2].isdigit() else (problem.file or name)
            found.add("Q-00", check, source, problem.count or 1, problem.total or None, action, problem.message)

    info_ls, osv_ls = info["ls"].unique(), osv["ls"].unique()
    first_month, last_month = osv["month"].min(), osv["month"].max()
    period_end = last_month + pd.offsets.MonthEnd(1)

    # Q-01 – Q-04. ЛС, которых нет в одном из файлов.
    only_info = np.setdiff1d(info_ls, osv_ls)
    only_osv = np.setdiff1d(osv_ls, info_ls)
    found.add("Q-01", "ЛС есть в анкете (01), но нет в ОСВ (02)", "01", len(only_info), len(info_ls), KEPT,
              "в панели таких ЛС нет: нет сальдо", _ls_examples(only_info))
    found.add("Q-02", "ЛС есть в ОСВ (02), но нет в анкете (01)", "02", len(only_osv), len(osv_ls), KEPT,
              "в ls_info для них флаги пустые", _ls_examples(only_osv))
    pay_ls = payments["ls"].unique()
    pay_orphans = np.setdiff1d(pay_ls, osv_ls)
    found.add("Q-03", "ЛС из оплат (03), которых нет в ОСВ", "03", len(pay_orphans), len(pay_ls), KEPT,
              f"{validator.number(int(payments['ls'].isin(pay_orphans).sum()))} строк оплат в панель не входят",
              _ls_examples(pay_orphans))
    event_ls = events["ls"].unique()
    event_orphans = np.setdiff1d(event_ls, osv_ls)
    found.add("Q-04", "ЛС из мер (04–13), которых нет в ОСВ", "04–13", len(event_orphans), len(event_ls), KEPT,
              "в журнале мер у них пустой dz_at_event", _ls_examples(event_orphans))

    # Q-05 – Q-07. Отрицательные суммы.
    negative = osv[osv["saldo_start"] < 0]
    found.add("Q-05", "Отрицательное сальдо на начало месяца (переплата)", "02", len(negative),
              int(osv["saldo_start"].notna().sum()), KEPT,
              f"{validator.number(negative['ls'].nunique())} ЛС; в долг не входит: dz = max(0, saldo_start)",
              negative.nsmallest(found.limit, "saldo_start").assign(
                  when=lambda d: _month(d["month"]), value=lambda d: _money(d["saldo_start"])))
    negative = osv[osv["accrual"] < 0]
    found.add("Q-06", "Отрицательное начисление (перерасчёт)", "02", len(negative),
              int(osv["accrual"].notna().sum()), KEPT, f"{validator.number(negative['ls'].nunique())} ЛС",
              negative.nsmallest(found.limit, "accrual").assign(
                  when=lambda d: _month(d["month"]), value=lambda d: _money(d["accrual"])))
    negative = payments[payments["amount"] < 0]
    found.add("Q-07", "Отрицательная оплата (сторно)", "03", len(negative), len(payments), KEPT,
              "сторно уменьшает сумму оплат за месяц",
              negative.nsmallest(found.limit, "amount").assign(
                  when=lambda d: d["payment_date"].dt.strftime("%Y-%m-%d"), value=lambda d: _money(d["amount"])))

    # Q-08. Сальдо не сходится: СЗ(M+1) ≠ СЗ(M) + начислено(M) − оплачено(M).
    ordered = osv.sort_values(["ls", "month"])
    following = ordered.groupby("ls")["saldo_start"].shift(-1)
    expected = ordered["saldo_start"] + ordered["accrual"] - ordered["paid"]
    known = following.notna() & expected.notna()
    gap = (following - expected).where(known)
    mismatch = gap.abs() > float(options["reconcile_tolerance"])
    sample = ordered.loc[mismatch].assign(gap=gap[mismatch], expected=expected[mismatch], actual=following[mismatch])
    sample = sample.reindex(sample["gap"].abs().sort_values(ascending=False).index)
    found.add("Q-08", "Сальдо не сходится между соседними месяцами", "02", int(mismatch.sum()), int(known.sum()), KEPT,
              "СЗ(M+1) ≠ СЗ(M) + начислено − оплачено: обычно перерасчёт; "
              f"{validator.number(sample['ls'].nunique())} ЛС",
              sample.assign(
                  when=lambda d: _month(d["month"] + pd.DateOffset(months=1)),
                  value=lambda d: "ожидалось " + _money(d["expected"]) + ", в файле " + _money(d["actual"])))

    # Q-09, Q-10. Даты в будущем — ошибка выгрузки, такие строки убираем.
    when = f"позже {as_of:%d.%m.%Y} (дата запуска или quality.as_of)"
    future = payments["payment_date"] > as_of
    found.add("Q-09", "Оплаты с датой в будущем", "03", int(future.sum()), len(payments), DROPPED, when,
              payments[future].assign(when=lambda d: d["payment_date"].dt.strftime("%Y-%m-%d"),
                                      value=lambda d: _money(d["amount"])))
    payments = _with_attrs(payments.loc[~future].reset_index(drop=True), raw["payments"])
    future = events["event_date"] > as_of
    found.add("Q-10", "Меры с датой в будущем", "04–13", int(future.sum()), len(events), DROPPED, when,
              events[future].assign(when=lambda d: d["event_date"].dt.strftime("%Y-%m-%d"),
                                    value=lambda d: d["measure"].astype(str)))
    events = _with_attrs(events.loc[~future].reset_index(drop=True), raw["events"])

    # Q-11, Q-12. Даты вне периода ОСВ: остаются, но в панель и результаты мер не попадают.
    period = f"период ОСВ {first_month:%Y-%m}…{last_month:%Y-%m}"
    outside = (payments["payment_date"] < first_month) | (payments["payment_date"] > period_end)
    found.add("Q-11", "Оплаты вне периода ОСВ", "03", int(outside.sum()), len(payments), KEPT,
              f"{period}; в оплаты панели не входят",
              payments[outside].assign(when=lambda d: d["payment_date"].dt.strftime("%Y-%m-%d"),
                                       value=lambda d: _money(d["amount"])))
    outside = (events["event_date"] < first_month) | (events["event_date"] > period_end)
    found.add("Q-12", "Меры вне периода ОСВ", "04–13", int(outside.sum()), len(events), KEPT,
              f"{period}; долг на месяц меры неизвестен",
              events[outside].assign(when=lambda d: d["event_date"].dt.strftime("%Y-%m-%d"),
                                     value=lambda d: d["measure"].astype(str)))

    # Q-13, Q-14. ЛС без движения во всей ОСВ.
    moves = osv[["saldo_start", "accrual", "paid"]].fillna(0).abs().sum(axis=1)
    still = moves.groupby(osv["ls"]).sum() == 0
    still_ls = still.index[still].to_numpy()
    with_events = np.intersect1d(still_ls, event_ls)
    inactive = np.setdiff1d(np.setdiff1d(still_ls, with_events), payments["ls"].unique())
    drop = bool(options["drop_inactive_ls"])
    found.add("Q-13", "ЛС без движения: нулевые сальдо, начисления и оплаты, нет мер", "02", len(inactive),
              len(osv_ls), DROPPED if drop else KEPT,
              ("убраны из ОСВ и анкеты, в панель не входят" if drop else "оставлены (quality.drop_inactive_ls: false)")
              + f"; {period}", _ls_examples(inactive))
    measures = events[events["ls"].isin(with_events)].groupby("ls")["measure"].agg(
        lambda m: ", ".join(sorted(set(m.astype(str))))).rename("value").reset_index()
    found.add("Q-14", "Меры у ЛС без движения в ОСВ", "02, 04–13", len(with_events), len(still_ls), KEPT,
              "у ЛС за весь период нет ни долга, ни начислений, а меры есть — проверьте номера ЛС в файлах мер",
              measures.astype({"ls": "Int64"}))
    if drop and len(inactive):
        osv = _with_attrs(osv[~osv["ls"].isin(inactive)].reset_index(drop=True), raw["osv"])
        info = _with_attrs(info[~info["ls"].isin(inactive)].reset_index(drop=True), raw["info"])

    # Q-15 – Q-17. Повторы и пропуски, которые не мешают расчётам.
    per_month = events.assign(month=events["event_date"].dt.to_period("M")).groupby(
        ["ls", "measure", "month"], observed=True).size()
    extra = (per_month - 1).groupby(level="measure", observed=True).sum()
    extra = extra[extra > 0].reindex(list(MEASURE_FILES)).dropna().astype(int)
    repeated = per_month[per_month > 1].reset_index(name="n")
    found.add("Q-15", "Повторы меры у ЛС в одном месяце", "04–13", int(extra.sum()), len(events),
              "свёрнуто в одну строку журнала мер (n_raw)",
              "; ".join(f"{m}: {validator.number(n)}" for m, n in extra.items()),
              repeated.nlargest(found.limit, "n").assign(
                  when=lambda d: d["month"].astype(str), value=lambda d: d["measure"].astype(str) + " × " + d["n"].astype(str)))
    same = payments.duplicated(keep=False)
    found.add("Q-16", "Одинаковые строки оплат (ЛС, дата, сумма, способ)", "03",
              int(payments.duplicated().sum()), len(payments), KEPT,
              "считаются отдельными платежами: ОСВ сходится вместе с ними",
              payments[same].assign(when=lambda d: d["payment_date"].dt.strftime("%Y-%m-%d"),
                                    value=lambda d: _money(d["amount"])))
    found.add("Q-17", "Не указан способ оплаты (NULL)", "03", int(payments["payment_method"].isna().sum()),
              len(payments), KEPT, "на расчёты не влияет")

    cleaned = {**raw, "info": info, "osv": osv, "payments": payments, "events": events}
    report = found.report(as_of)
    _log(report)
    return cleaned, report


def _log(report: QualityReport) -> None:
    checks = report.summary[report.summary["code"] != "Q-00"]
    dropped = checks[checks["action"].str.startswith(DROPPED) & (checks["count"] > 0)]
    for row in dropped.itertuples(index=False):
        logger.warning("Очистка: %s — %s, %s.", row.check, validator.number(int(row.count)), row.action)
    noted = checks[~checks["action"].str.startswith(DROPPED) & (checks["count"] > 0)]
    logger.info(
        "Отчёт о качестве: %d проверок, замечания по %d, строк убрано по %d (даты на %s).",
        len(checks), len(noted) + len(dropped), len(dropped), f"{report.as_of:%d.%m.%Y}",
    )


def write_report(report: QualityReport, folder: Path, meta: dict[str, str] | None = None) -> Path:
    """Пишет отчёт в folder/quality_report.xlsx: листы «Сводка», «Примеры», «Запуск»."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / REPORT_FILE
    summary = report.summary.rename(columns=SUMMARY_HEADERS)
    examples = report.examples.rename(columns=EXAMPLE_HEADERS)
    run = pd.DataFrame(
        list({**(meta or {}), "Даты в будущем считаются от": f"{report.as_of:%d.%m.%Y}",
              "Сформирован": f"{pd.Timestamp.now():%d.%m.%Y %H:%M}"}.items()),
        columns=["Параметр", "Значение"],
    )
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet, frame, widths in (
                ("Сводка", summary, [7, 55, 12, 12, 12, 8, 28, 90]),
                ("Примеры", examples, [7, 12, 14, 40, 60]),
                ("Запуск", run, [32, 60]),
            ):
                frame.to_excel(writer, sheet_name=sheet, index=False)
                sheet_obj = writer.sheets[sheet]
                for column, width in zip("ABCDEFGH", widths):
                    sheet_obj.column_dimensions[column].width = width
                sheet_obj.freeze_panes = "A2"
            for cell in writer.sheets["Сводка"]["F"][1:]:
                cell.number_format = "0.0%"
    except PermissionError as error:
        raise OSError(f"Не удалось записать «{path}»: файл открыт в Excel? Закройте его и запустите снова.") from error
    return path
