"""Журнал запуска и сообщения об ошибках (SVC-6, tns/logs.py).

Главная проверка — сценарии поломок: для каждого run.py должен сказать, что не так
(называя файл, колонку или параметр) и что сделать, и показать путь к журналу.
"""
from __future__ import annotations

import errno
import logging
import re
import sys
import types

import pytest
import yaml

from tns import logs, pipeline
from tns.data import loaders, overdue_check, panel, panel_check
from tests.conftest import write_payments, write_xlsx

# Слова, с которых начинается совет пользователю.
HINT = re.compile(r"Укажите|Исправьте|Положите|Закройте|Проверьте|Выгрузите|Пришлите|Установите|Пересохраните"
                  r"|Откройте|Добавьте|Верните|Освободите|Оставьте|постройте|запустите снова", re.IGNORECASE)


@pytest.fixture
def config(settings, tmp_path):
    settings["paths"] = {**settings["paths"], "cache_dir": str(tmp_path / "cache")}
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")
    return path


def run(data_dir, tmp_path, config, *extra, month="2025-02"):
    args = ["--data", str(data_dir), "--out", str(tmp_path / "out"), "--config", str(config), "--no-cache"]
    return pipeline.main((["--month", month] if month else []) + args + list(extra))


def error_text(caplog) -> str:
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1, errors
    return errors[0]


def fake_step(monkeypatch, function):
    module = types.ModuleType("tns.rules.stages")
    module.build_stages = function
    monkeypatch.setitem(sys.modules, "tns.rules.stages", module)


# ---------------------------------------------------------------------------
# Сценарии поломок: что не так + что сделать
# ---------------------------------------------------------------------------

def break_settings(config, change):
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    change(data)
    config.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")


def _raise(error):
    def step(ctx):
        raise error
    return step


SCENARIOS = {
    # имя: (как сломать (data_dir, tmp_path, config, monkeypatch) → доп. аргументы, что должно быть в сообщении)
    "нет месяца": (lambda d, t, c, m: {"month": None}, "Не указан месяц плана"),
    "месяц в неверном формате": (lambda d, t, c, m: {"month": "март"}, "Неверный месяц «март»"),
    "месяца нет в данных": (lambda d, t, c, m: {"month": "2026-05"}, "в данных нет месяца 2026-05"),
    "нет папки с данными": (lambda d, t, c, m: {"data": t / "нет"}, "Не найдена папка с данными"),
    "нет файла 06": (lambda d, t, c, m: (d / loaders.MEASURE_FILES["sms"]).unlink(), "Не найден входной файл «06 СМС ХК.xlsx»"),
    "в файле 06 нет колонки Дата": (
        lambda d, t, c, m: loaders_write_measure_without_date(d), "В файле «06 СМС ХК.xlsx» нет колонки «Дата»"),
    "пустой файл оплат": (lambda d, t, c, m: write_payments(d / loaders.PAYMENTS_FILE, []), "только шапка"),
    "файл 02 повреждён": (
        lambda d, t, c, m: (d / loaders.OSV_FILE).write_bytes(b"not an excel file"), "не открывается как книга Excel"),
    "нет файла настроек": (lambda d, t, c, m: {"config": t / "нет.yaml"}, "Не найден файл настроек"),
    "настройки — сломанный YAML": (
        lambda d, t, c, m: c.write_text("result:\n  share: [0.3\n", encoding="utf-8"), "файл настроек не читается, строка"),
    "в настройках нет раздела": (lambda d, t, c, m: break_settings(c, lambda s: s.pop("result")), "не найден параметр или колонка «result»"),
    "неверная доля в настройках": (
        lambda d, t, c, m: break_settings(c, lambda s: s["result"].update(share=30)), "result.share — доля долга от 0 до 1"),
    "--out указывает на файл": (
        lambda d, t, c, m: (t / "out").write_text("x"), "лежит файл, а нужна папка"),
    "ошибка в коде шага": (lambda d, t, c, m: fake_step(m, lambda ctx: 1 / 0), "внутренняя ошибка программы (ZeroDivisionError"),
    "не установлена библиотека": (
        lambda d, t, c, m: fake_step(m, _raise(ModuleNotFoundError("No module named 'catboost'", name="catboost"))),
        "не установлена библиотека «catboost»"),
    "не хватает памяти": (lambda d, t, c, m: fake_step(m, _raise(MemoryError())), "не хватает оперативной памяти"),
    "нет доступа к файлу": (
        lambda d, t, c, m: fake_step(m, _raise(PermissionError(errno.EACCES, "Permission denied", "plan.xlsx"))),
        "нет доступа к «plan.xlsx»"),
    "закончилось место на диске": (
        lambda d, t, c, m: fake_step(m, _raise(OSError(errno.ENOSPC, "No space left on device"))), "на диске закончилось место"),
}


def loaders_write_measure_without_date(folder):
    write_xlsx(folder / loaders.MEASURE_FILES["sms"], [["СМС", None], ["ЛС", "Дат"], [10, "2025-01-01"]])


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_every_error_says_what_and_what_to_do(name, data_dir, tmp_path, config, monkeypatch, caplog, log_dir):
    breaker, what = SCENARIOS[name]
    extra = breaker(data_dir, tmp_path, config, monkeypatch)
    extra = extra if isinstance(extra, dict) else {}
    args = []
    if "data" in extra:
        data_dir = extra["data"]
    if "config" in extra:
        config = extra["config"]
    month = extra.get("month", "2025-02")
    with caplog.at_level(logging.DEBUG):
        assert run(data_dir, tmp_path, config, *args, month=month) == 1
    text = error_text(caplog)
    assert what in text, text
    assert HINT.search(text), f"нет совета, что делать: {text}"
    assert f"Подробности — в журнале «{log_dir}" in text
    # На экране — без трассировки Python: она только в журнале (уровень DEBUG).
    assert "Traceback" not in text
    assert all(r.levelno == logging.DEBUG for r in caplog.records if r.exc_info)


def test_keyboard_interrupt(data_dir, tmp_path, config, monkeypatch, caplog):
    fake_step(monkeypatch, _raise(KeyboardInterrupt()))
    assert run(data_dir, tmp_path, config) == 1
    assert "Запуск прерван (Ctrl+C)" in error_text(caplog) and "запустите снова" in error_text(caplog)


@pytest.mark.parametrize("main, args, what", [
    (panel.main, ["--data", "нет такой папки"], "Не найдена папка с данными"),
    (loaders.main, ["--data", "нет такой папки"], "Не найдена папка с данными"),
    (overdue_check.main, ["--month", "2025-02", "--out", "нет такой папки"], "Сначала постройте панель"),
    (panel_check.main, ["--month", "2025-02", "--out", "нет такой папки"], "Сначала постройте панель"),
])
def test_other_commands_explain_errors(main, args, what, caplog, log_dir):
    with caplog.at_level(logging.INFO):
        assert main(args) == 1
    text = error_text(caplog)
    assert what in text and HINT.search(text) and f"журнале «{log_dir}" in text


# ---------------------------------------------------------------------------
# explain: подсказка для каждой непредвиденной ошибки
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("error", [
    PermissionError(errno.EACCES, "Permission denied", "a.xlsx"),
    OSError(errno.ENOSPC, "No space left on device"),
    FileNotFoundError(errno.ENOENT, "No such file or directory", "x.xlsx"),
    NotADirectoryError(errno.ENOTDIR, "Not a directory", "out"),
    OSError(errno.EIO, "Input/output error", "data.xlsx"),
    MemoryError(),
    ModuleNotFoundError("No module named 'lightgbm'", name="lightgbm"),
    KeyError("result"),
    RuntimeError("что-то сломалось"),
    ZeroDivisionError("division by zero"),
])
def test_unexpected_errors_get_a_hint(error):
    message, hint = logs.explain(error)
    assert message and hint and HINT.search(hint)


def test_own_messages_are_kept():
    # Сообщения программы уже говорят, что сделать: explain их не переписывает.
    error = FileNotFoundError("Не найдена папка с данными «x». Укажите папку с файлами 01–14 в --data.")
    assert logs.explain(error) == (str(error), None)


# ---------------------------------------------------------------------------
# Журнал в файле
# ---------------------------------------------------------------------------

class TestLogFile:
    def test_successful_run_writes_log(self, data_dir, tmp_path, config, log_dir, capsys):
        assert run(data_dir, tmp_path, config) == 0
        files = list(log_dir.glob("run_*.log"))
        assert len(files) == 1
        text = files[0].read_text(encoding="utf-8")
        assert "Команда: run --month 2025-02" in text and "pandas" in text
        assert "Шаг 4/9: панель «ЛС × месяц» — готово" in text
        assert f"Журнал запуска: «{files[0]}»" in capsys.readouterr().err

    def test_error_traceback_goes_to_file(self, data_dir, tmp_path, config, monkeypatch, log_dir, capsys):
        fake_step(monkeypatch, lambda ctx: 1 / 0)
        assert run(data_dir, tmp_path, config) == 1
        text = next(log_dir.glob("run_*.log")).read_text(encoding="utf-8")
        assert "Трассировка ошибки" in text and "ZeroDivisionError" in text and "Traceback" in text
        screen = capsys.readouterr().err
        assert "ОШИБКА Шаг 5 «стадии»: внутренняя ошибка программы" in screen and "Traceback" not in screen

    def test_console_labels(self, capsys):
        logs.setup_logging("test")
        logging.getLogger("tns.x").info("готово")
        logging.getLogger("tns.x").warning("мало данных")
        err = capsys.readouterr().err
        assert re.search(r"\d\d:\d\d:\d\d готово", err) and re.search(r"\d\d:\d\d:\d\d ВНИМАНИЕ мало данных", err)

    def test_old_logs_are_removed(self, log_dir, monkeypatch):
        log_dir.mkdir()
        for day in range(1, 6):
            (log_dir / f"run_2026-01-0{day}_10-00-00.log").write_text("")
        (log_dir / "panel_2026-01-01_10-00-00.log").write_text("")
        monkeypatch.setattr(logs, "_settings_value", lambda s, section, key, default: 3 if key == "keep" else default)
        path = logs.setup_logging("run")
        kept = sorted(p.name for p in log_dir.glob("run_*.log"))
        assert len(kept) == 3 and path.name in kept and "run_2026-01-05_10-00-00.log" in kept
        assert (log_dir / "panel_2026-01-01_10-00-00.log").exists()  # журналы других команд не трогаем

    def test_unwritable_log_folder_is_not_fatal(self, tmp_path, monkeypatch, caplog):
        blocker = tmp_path / "файл"
        blocker.write_text("")
        monkeypatch.setenv("TNS_LOG_DIR", str(blocker / "logs"))
        with caplog.at_level(logging.WARNING):
            assert logs.setup_logging("run") is None
        assert "Журнал в файл не пишется" in caplog.text and "Проверьте, что папка" in caplog.text


def test_run_py_without_library():
    """Если не установлена библиотека, run.py объясняет, что сделать, а не падает с трассировкой."""
    import subprocess
    code = ("import sys, runpy; sys.modules['pandas'] = None; sys.argv = ['run.py', '--month', '2026-02']; "
            f"runpy.run_path({str(loaders.REPO_ROOT / 'run.py')!r}, run_name='__main__')")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=loaders.REPO_ROOT)
    assert done.returncode == 1
    assert "Не установлена библиотека «pandas»" in done.stderr and "pip install -r requirements.txt" in done.stderr
    assert "Traceback" not in done.stderr
