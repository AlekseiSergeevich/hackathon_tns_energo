"""Тесты единой точки запуска run.py / tns/pipeline.py (SVC-4) на маленьких файлах из conftest."""
from __future__ import annotations

import logging
import subprocess
import sys
import types

import pandas as pd
import pytest
import yaml

from tns import pipeline
from tns.data import loaders


@pytest.fixture
def config(settings, tmp_path):
    """Файл настроек для --config: кэш во временной папке, запасные лимиты из фикстуры settings."""
    settings["paths"] = {**settings["paths"], "cache_dir": str(tmp_path / "cache")}
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")
    return path


def run(data_dir, out, config, month="2025-02", *extra):
    return pipeline.main(["--month", month, "--data", str(data_dir), "--out", str(out), "--config", str(config), *extra])


def fake_module(monkeypatch, name, **functions):
    """Подкладывает модуль шага (например, tns.rules.stages), как будто его уже написали."""
    module = types.ModuleType(name)
    for function_name, function in functions.items():
        setattr(module, function_name, function)
    monkeypatch.setitem(sys.modules, name, module)


class TestCommandLine:
    def test_data_steps_write_tables(self, data_dir, tmp_path, config, caplog):
        with caplog.at_level(logging.INFO):
            assert run(data_dir, tmp_path / "out", config) == 0
        month_dir = tmp_path / "out" / "2025-02"
        assert sorted(p.name for p in month_dir.iterdir()) == [
            "ls_info.parquet", "measure_events.parquet", "panel.parquet",
            "quality.parquet", "quality_examples.parquet", "quality_report.xlsx",
        ]
        assert "Шаг 5/9: стадии — пока не реализован" in caplog.text
        assert "План на 2025-02 не построен" in caplog.text

    def test_written_tables_follow_contract(self, data_dir, tmp_path, config):
        run(data_dir, tmp_path / "out", config)
        panel = pd.read_parquet(tmp_path / "out" / "2025-02" / "panel.parquet")
        assert str(panel["month"].dtype) == "datetime64[ns]" and str(panel["ls"].dtype) == "int64"

    def test_bad_month_format(self, data_dir, tmp_path, config, caplog):
        assert run(data_dir, tmp_path / "out", config, "02.2025") == 1
        assert "укажите месяц в формате ГГГГ-ММ" in caplog.text
        assert not (tmp_path / "out").exists()

    def test_month_outside_data(self, data_dir, tmp_path, config, caplog):
        assert run(data_dir, tmp_path / "out", config, "2026-05") == 1
        assert "Шаг 2 «проверка данных»: в данных нет месяца 2026-05" in caplog.text
        assert "2025-01…2025-03" in caplog.text

    def test_month_is_required(self, data_dir, config, caplog):
        assert pipeline.main(["--data", str(data_dir), "--config", str(config)]) == 1
        assert "Не указан месяц плана. Добавьте --month" in caplog.text

    def test_missing_data_folder(self, tmp_path, config, caplog):
        assert run(tmp_path / "нет", tmp_path / "out", config) == 1
        assert "Не найдена папка с данными" in caplog.text

    def test_broken_input_file_names_step(self, data_dir, tmp_path, config, caplog):
        (data_dir / loaders.MEASURE_FILES["sms"]).unlink()
        assert run(data_dir, tmp_path / "out", config, "2025-02", "--no-cache") == 1
        assert "Шаг 1 «загрузка файлов 01–14»" in caplog.text and "06 СМС ХК.xlsx" in caplog.text

    def test_run_py_from_shell(self, data_dir, tmp_path, config):
        done = subprocess.run(
            [sys.executable, str(loaders.REPO_ROOT / "run.py"), "--month", "2025-02", "--data", str(data_dir),
             "--out", str(tmp_path / "out"), "--config", str(config)],
            capture_output=True, text=True, encoding="utf-8", cwd=tmp_path,
        )
        assert done.returncode == 0, done.stderr
        assert "Шаг 9/9" in done.stderr


class TestSteps:
    """Подключение шагов 4–8 по именам «модуль:функция»."""

    @pytest.fixture
    def ctx(self, data_dir, tmp_path, settings):
        return pipeline.RunContext(
            month=pd.Timestamp("2025-02-01"), data_dir=data_dir, out_dir=tmp_path / "out" / "2025-02",
            rules_path=tmp_path / "rules.xlsx", settings=settings,
        )

    def test_full_chain_builds_plan(self, ctx, monkeypatch, caplog):
        def stages(c):
            return c.tables["panel"][["ls", "month"]].assign(stage="S1")

        def candidates(c):
            return c.tables["stages"].query("month == @c.month")[["ls", "month"]].assign(measure="sms")

        def plan(c):
            assert "score" in c.tables and len(c.raw["limits"]) == 10
            return c.tables["candidates"].assign(status="assigned")

        def registries(c):
            (c.out_dir / "sms.xlsx").write_text("реестр", encoding="utf-8")

        fake_module(monkeypatch, "tns.rules.stages", build_stages=stages)
        fake_module(monkeypatch, "tns.rules.engine", build_candidates=candidates)
        fake_module(monkeypatch, "tns.ml.score", build_score=lambda c: pd.DataFrame({"ls": [10], "p_pay": [0.5]}))
        fake_module(monkeypatch, "tns.alloc.allocator", build_plan=plan)
        fake_module(monkeypatch, "tns.alloc.registries", write_registries=registries)

        results = pipeline.run_pipeline(ctx)
        assert [r.status for r in results] == [pipeline.DONE] * 9
        for name in ("stages", "candidates", "score", "plan"):
            assert (ctx.out_dir / f"{name}.parquet").exists()
        assert (ctx.out_dir / "sms.xlsx").exists()
        assert len(pd.read_parquet(ctx.out_dir / "plan.parquet")) == 3

    def test_missing_score_does_not_block_plan(self, ctx, monkeypatch):
        fake_module(monkeypatch, "tns.rules.stages", build_stages=lambda c: pd.DataFrame({"ls": [10]}))
        fake_module(monkeypatch, "tns.rules.engine", build_candidates=lambda c: pd.DataFrame({"ls": [10]}))
        fake_module(monkeypatch, "tns.alloc.allocator", build_plan=lambda c: pd.DataFrame({"ls": [10]}))
        statuses = {r.title: r.status for r in pipeline.run_pipeline(ctx)}
        assert statuses["скор"] == pipeline.NOT_IMPLEMENTED
        assert statuses["аллокатор под лимиты"] == pipeline.DONE
        assert statuses["реестры по мерам"] == pipeline.NOT_IMPLEMENTED

    def test_module_without_function_is_not_implemented(self, ctx, monkeypatch):
        fake_module(monkeypatch, "tns.rules.stages")
        assert pipeline.run_pipeline(ctx)[4].status == pipeline.NOT_IMPLEMENTED

    def test_not_implemented_error_is_not_implemented(self, ctx, monkeypatch):
        def stages(c):
            raise NotImplementedError
        fake_module(monkeypatch, "tns.rules.stages", build_stages=stages)
        assert pipeline.run_pipeline(ctx)[4].status == pipeline.NOT_IMPLEMENTED

    def test_step_error_names_step(self, ctx, monkeypatch):
        def stages(c):
            raise ValueError("в листе «Стадии» нет S0")
        fake_module(monkeypatch, "tns.rules.stages", build_stages=stages)
        with pytest.raises(pipeline.StepError, match="Шаг 5 «стадии»: в листе «Стадии» нет S0"):
            pipeline.run_pipeline(ctx)

    def test_wrong_tables_returned(self, ctx, monkeypatch):
        fake_module(monkeypatch, "tns.rules.stages", build_stages=lambda c: {"стадии": pd.DataFrame()})
        with pytest.raises(pipeline.StepError, match="должен вернуть таблицы stages"):
            pipeline.run_pipeline(ctx)

    def test_missing_library_inside_step_is_an_error(self, ctx, monkeypatch):
        def broken(name, *args, **kwargs):
            if name == "tns.rules.stages":
                raise ModuleNotFoundError("No module named 'catboost'", name="catboost")
            return original(name, *args, **kwargs)
        original = pipeline.importlib.import_module
        monkeypatch.setattr(pipeline.importlib, "import_module", broken)
        with pytest.raises(pipeline.StepError, match="catboost"):
            pipeline.run_pipeline(ctx)

    def test_internal_error_points_to_log(self, data_dir, tmp_path, config, monkeypatch, caplog):
        def stages(c):
            return 1 / 0
        fake_module(monkeypatch, "tns.rules.stages", build_stages=stages)
        assert run(data_dir, tmp_path / "out", config) == 1
        assert "Шаг 5 «стадии»: внутренняя ошибка программы (ZeroDivisionError: division by zero)" in caplog.text
        assert "Что сделать: Это не ошибка в ваших данных. Пришлите файл журнала разработчикам." in caplog.text
        # Трассировка — в журнале (уровень DEBUG), не на экране.
        traceback = [r for r in caplog.records if r.exc_info]
        assert traceback and all(r.levelno == logging.DEBUG for r in traceback)
