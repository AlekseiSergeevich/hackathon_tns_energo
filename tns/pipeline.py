"""План мер на месяц: девять шагов от файлов 01–14 до реестров.

    1 загрузка файлов 01–14     tns/data/loaders.py (с проверкой, tns/data/validator.py)
    2 проверка данных           здесь
    3 очистка и отчёт о качестве tns/data/quality.py           → quality, quality_examples, quality_report.xlsx
    4 панель «ЛС × месяц»       tns/data/panel.py              → ls_info, panel, measure_events
    5 стадии                    tns.rules.stages:build_stages  → stages
    6 кандидаты на меры         tns.rules.engine:build_candidates → candidates
    7 скор                      tns.ml.score:build_score       → score
    8 аллокатор под лимиты      tns.alloc.allocator:build_plan → plan
    9 реестры по мерам          tns.alloc.registries:write_registries (файлы Excel в папку месяца)

Шаги 5–9 подключаются по имени «модуль:функция». Функция получает RunContext и возвращает
таблицу (DataFrame), если шаг строит одну таблицу, словарь {имя: DataFrame} или None.
Пока модуля или функции нет, шаг считается нереализованным, и пайплайн идёт дальше.
Шаг, которому не хватает входных таблиц, пропускается.

Таблицы пишутся в output/<YYYY-MM>/<имя>.parquet (docs/interfaces.md, «Общие соглашения»).
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from tns.data import panel as panel_module
from tns.data import quality
from tns.data.loaders import REPO_ROOT, default_cache_dir, load_all, load_settings

logger = logging.getLogger(__name__)

DONE, NOT_IMPLEMENTED, SKIPPED = "выполнен", "не реализован", "пропущен"


@dataclass
class RunContext:
    """Всё, что нужно шагам: параметры запуска и уже построенные таблицы."""
    month: pd.Timestamp
    data_dir: Path
    out_dir: Path  # папка месяца: output/<YYYY-MM>
    rules_path: Path
    settings: dict
    cache_dir: Path | None = None
    raw: dict[str, pd.DataFrame] = field(default_factory=dict)  # load_all: info, osv, payments, events, limits
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)  # таблицы docs/interfaces.md


@dataclass(frozen=True)
class Step:
    title: str
    target: Callable[[RunContext], object] | str
    needs: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()


@dataclass
class StepResult:
    number: int
    title: str
    status: str
    seconds: float = 0.0
    note: str = ""


class StepError(Exception):
    """Шаг упал. В сообщении — номер и название шага; исходная ошибка в __cause__."""

    def __init__(self, number: int, step: Step, error: BaseException):
        super().__init__(f"Шаг {number} «{step.title}»: {error}")
        self.number, self.step = number, step


# ---------------------------------------------------------------------------
# Шаги 1–3: данные
# ---------------------------------------------------------------------------

def _load(ctx: RunContext) -> None:
    ctx.raw = load_all(ctx.data_dir, ctx.settings, ctx.cache_dir)


def _validate(ctx: RunContext) -> None:
    months = ctx.raw["osv"]["month"]
    first, last = months.min(), months.max()
    if not first <= ctx.month <= last:
        raise ValueError(
            f"в данных нет месяца {ctx.month:%Y-%m}: ОСВ покрывает {first:%Y-%m}…{last:%Y-%m}. "
            "Укажите месяц из этого периода или положите в папку данных выгрузку с нужным месяцем."
        )
    # Свежие предупреждения загрузчики уже вывели в лог; из кэша — повторяем здесь.
    warnings = [p for table in ctx.raw.values() for p in table.attrs.get("problems", [])]
    for table in ctx.raw.values():
        if table.attrs.get("cached"):
            for problem in table.attrs.get("problems", []):
                logger.warning("%s", problem.message)
    if warnings:
        logger.info("Предупреждений по входным файлам: %d. Запуск продолжается.", len(warnings))
    if not ctx.rules_path.exists():
        logger.warning("Не найдена таблица правил «%s»: без неё не построить стадии и кандидатов.", ctx.rules_path)


def _clean(ctx: RunContext) -> dict[str, pd.DataFrame]:
    ctx.raw, report = quality.clean(ctx.raw, ctx.settings)
    path = quality.write_report(report, ctx.out_dir, {
        "Месяц плана": f"{ctx.month:%Y-%m}", "Папка с данными": str(ctx.data_dir),
    })
    logger.info("Отчёт о качестве данных: «%s».", path)
    return {"quality": report.summary, "quality_examples": report.examples}


def _build_panel(ctx: RunContext) -> dict[str, pd.DataFrame]:
    return panel_module.build_from_raw(ctx.raw, ctx.settings)


STEPS: list[Step] = [
    Step("загрузка файлов 01–14", _load),
    Step("проверка данных", _validate),
    Step("очистка данных и отчёт о качестве", _clean, produces=("quality", "quality_examples")),
    Step("панель «ЛС × месяц»", _build_panel, produces=("ls_info", "panel", "measure_events")),
    Step("стадии", "tns.rules.stages:build_stages",
         needs=("ls_info", "panel", "measure_events"), produces=("stages",)),
    Step("кандидаты на меры", "tns.rules.engine:build_candidates",
         needs=("ls_info", "panel", "measure_events", "stages"), produces=("candidates",)),
    Step("скор", "tns.ml.score:build_score", needs=("ls_info", "panel", "stages"), produces=("score",)),
    Step("аллокатор под лимиты", "tns.alloc.allocator:build_plan", needs=("candidates",), produces=("plan",)),
    Step("реестры по мерам", "tns.alloc.registries:write_registries", needs=("plan",)),
]


# ---------------------------------------------------------------------------
# Запуск шагов
# ---------------------------------------------------------------------------

def _resolve(target: Callable | str) -> Callable | None:
    """Функция шага или None, если её модуль или функция ещё не написаны."""
    if callable(target):
        return target
    module_name, function_name = target.split(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        # Нет самого модуля шага — шаг не реализован. Если внутри модуля не хватает
        # другой библиотеки, это настоящая ошибка, её показываем.
        if error.name and (module_name == error.name or module_name.startswith(error.name + ".")):
            return None
        raise
    return getattr(module, function_name, None)


def _collect(step: Step, result: object) -> dict[str, pd.DataFrame]:
    if result is None:
        tables = {}
    elif isinstance(result, pd.DataFrame) and len(step.produces) == 1:
        tables = {step.produces[0]: result}
    elif isinstance(result, dict):
        tables = result
    else:
        raise TypeError(f"шаг вернул {type(result).__name__}, а нужен DataFrame, словарь таблиц или None")
    if set(tables) != set(step.produces):
        raise ValueError(
            f"шаг должен вернуть таблицы {', '.join(step.produces) or '(никаких)'}, "
            f"а вернул {', '.join(tables) or '(никаких)'}"
        )
    for name, table in tables.items():
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f"таблица «{name}» — {type(table).__name__}, а нужен DataFrame")
    return tables


def run_pipeline(ctx: RunContext, steps: list[Step] | None = None) -> list[StepResult]:
    """Выполняет шаги по порядку и пишет их таблицы в ctx.out_dir. Ошибка шага — StepError."""
    steps = STEPS if steps is None else steps
    results = []
    for number, step in enumerate(steps, start=1):
        prefix = f"Шаг {number}/{len(steps)}: {step.title}"
        missing = [name for name in step.needs if name not in ctx.tables]
        if missing:
            logger.info("%s — пропущен: нет входных таблиц %s.", prefix, ", ".join(missing))
            results.append(StepResult(number, step.title, SKIPPED, note=f"нет таблиц {', '.join(missing)}"))
            continue
        try:
            function = _resolve(step.target)
            if function is None:
                logger.info("%s — пока не реализован (%s).", prefix, step.target)
                results.append(StepResult(number, step.title, NOT_IMPLEMENTED, note=str(step.target)))
                continue
            logger.info("%s…", prefix)
            started = time.perf_counter()
            try:
                tables = _collect(step, function(ctx))
            except NotImplementedError:
                logger.info("%s — пока не реализован (%s).", prefix, step.target)
                results.append(StepResult(number, step.title, NOT_IMPLEMENTED, note=str(step.target)))
                continue
            for name, table in tables.items():
                ctx.tables[name] = table
                ctx.out_dir.mkdir(parents=True, exist_ok=True)
                panel_module.write_parquet(table, ctx.out_dir / f"{name}.parquet")
        except Exception as error:
            raise StepError(number, step, error) from error
        seconds = time.perf_counter() - started
        written = "; ".join(f"{name}: {len(table):,} строк".replace(",", " ") for name, table in tables.items())
        logger.info("%s — готово за %.1f с%s.", prefix, seconds, f" ({written})" if written else "")
        results.append(StepResult(number, step.title, DONE, seconds, written))
    return results


def _report(results: list[StepResult], ctx: RunContext) -> None:
    done = [r for r in results if r.status == DONE]
    not_implemented = [r.title for r in results if r.status == NOT_IMPLEMENTED]
    skipped = [r.title for r in results if r.status == SKIPPED]
    logger.info("Итог: выполнено %d из %d шагов. Результаты в «%s».", len(done), len(results), ctx.out_dir)
    if not_implemented:
        logger.warning("Пока не реализованы: %s.", ", ".join(not_implemented))
    if skipped:
        logger.warning("Пропущены, потому что нет входных таблиц: %s.", ", ".join(skipped))
    if "plan" in ctx.tables:
        logger.info("План на %s: «%s».", f"{ctx.month:%Y-%m}", ctx.out_dir / "plan.parquet")
    else:
        logger.warning("План на %s не построен: не хватает шагов выше.", f"{ctx.month:%Y-%m}")


# ---------------------------------------------------------------------------
# Командная строка
# ---------------------------------------------------------------------------

def _resolve_path(value: str | None, default: str) -> Path:
    return Path(value) if value else REPO_ROOT / default


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Строит план мер воздействия на месяц: все шаги одной командой.",
        epilog='Пример: python run.py --month 2026-02 --data "Материалы к кейсу"',
    )
    parser.add_argument("--month", help="месяц плана ГГГГ-ММ, например 2026-02")
    parser.add_argument("--data", help="папка с файлами 01–14 (по умолчанию из configs/settings.yaml)")
    parser.add_argument("--out", help="папка результатов; план ляжет в <папка>/<ГГГГ-ММ> (по умолчанию output)")
    parser.add_argument("--rules", help="таблица правил .xlsx (по умолчанию из configs/settings.yaml)")
    parser.add_argument("--config", help="файл настроек (по умолчанию configs/settings.yaml)")
    parser.add_argument("--no-cache", action="store_true", help="читать исходные файлы заново, не используя кэш")
    args = parser.parse_args(argv)

    # В консоли Windows не все символы есть в кодировке: лучше \uXXXX, чем падение на выводе.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    try:
        if not args.month:
            raise ValueError("не указан месяц плана. Пример: python run.py --month 2026-02 --data <папка>")
        settings = load_settings(args.config)
        month = panel_module.parse_plan_month(args.month)
        data_dir = _resolve_path(args.data, settings["paths"]["data_dir"])
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Не найдена папка с данными: «{data_dir}». Укажите её в --data.")
        ctx = RunContext(
            month=month,
            data_dir=data_dir,
            out_dir=_resolve_path(args.out, settings["paths"]["output_dir"]) / f"{month:%Y-%m}",
            rules_path=_resolve_path(args.rules, settings["paths"]["rules"]),
            settings=settings,
            cache_dir=None if args.no_cache else default_cache_dir(settings),
        )
    except (OSError, ValueError, KeyError) as error:
        logger.error("Запуск невозможен: %s", error)
        return 1

    logger.info("План на %s. Данные: «%s».", f"{month:%Y-%m}", data_dir)
    try:
        results = run_pipeline(ctx)
    except StepError as error:
        if isinstance(error.__cause__, (OSError, ValueError, KeyError)):
            logger.error("%s", error)
        else:
            logger.error("%s\nВнутренняя ошибка программы. Пришлите этот вывод разработчикам.", error,
                         exc_info=error.__cause__)
        return 1
    _report(results, ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
