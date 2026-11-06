"""Журнал запуска и понятные сообщения об ошибках (SVC-6).

- setup_logging: короткие сообщения в консоль и подробный журнал в файл
  output/logs/<команда>_<дата_время>.log (папка — paths.log_dir в configs/settings.yaml,
  или переменная окружения TNS_LOG_DIR). Хранится не больше logs.keep последних журналов.
- explain: по любой ошибке — что не так и что сделать.
- report_error: пишет это в консоль и журнал, а трассировку Python — только в журнал.

Сообщения об ошибках данных и настроек формулируются там, где ошибка найдена, и сразу
говорят, что исправить. explain добавляет подсказку для ошибок, которые программа заранее
не предусмотрела (нет памяти, нет доступа к файлу, не установлена библиотека, сбой в коде).
"""
from __future__ import annotations

import errno
import logging
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import yaml

logger = logging.getLogger("tns")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = "output/logs"
DEFAULT_KEEP = 30
LEVEL_NAMES = {"WARNING": "ВНИМАНИЕ", "ERROR": "ОШИБКА", "CRITICAL": "ОШИБКА"}
REQUIREMENTS = "python -m pip install -r requirements.txt"


class _ConsoleFormatter(logging.Formatter):
    """«12:04:05 Шаг 1/9…», «12:04:05 ВНИМАНИЕ …», «12:04:05 ОШИБКА …»."""

    def format(self, record: logging.LogRecord) -> str:
        level = LEVEL_NAMES.get(record.levelname)
        time = self.formatTime(record, "%H:%M:%S")
        return f"{time} {level} {record.getMessage()}" if level else f"{time} {record.getMessage()}"


def _settings_value(settings: dict | None, section: str, key: str, default):
    if settings is None:
        try:
            with open(REPO_ROOT / "configs" / "settings.yaml", encoding="utf-8") as handle:
                settings = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError):
            settings = {}
    return (settings.get(section) or {}).get(key, default)


def log_dir(settings: dict | None = None) -> Path:
    """Папка журналов: TNS_LOG_DIR, иначе paths.log_dir из настроек, иначе output/logs."""
    value = os.environ.get("TNS_LOG_DIR") or _settings_value(settings, "paths", "log_dir", DEFAULT_LOG_DIR)
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def setup_logging(command: str, settings: dict | None = None, argv: list[str] | None = None) -> Path | None:
    """Настраивает журнал команды и возвращает путь к файлу (None, если файл создать не удалось)."""
    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, "_tns", False)]:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.DEBUG)

    # В консоли Windows не все символы есть в кодировке: лучше \uXXXX, чем падение на выводе.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="backslashreplace")
            except (ValueError, OSError):
                pass
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(_ConsoleFormatter())
    console._tns = True
    root.addHandler(console)

    folder = log_dir(settings)
    path = folder / f"{command}_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
    except OSError as error:
        logger.warning("Журнал в файл не пишется (%s): сообщения будут только на экране. "
                       "Проверьте, что папка «%s» доступна для записи.", error.strerror or error, folder)
        return None
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    file_handler._tns = True
    root.addHandler(file_handler)
    _remove_old(folder, command, int(_settings_value(settings, "logs", "keep", DEFAULT_KEEP)))

    logger.debug("Команда: %s %s", command, " ".join(sys.argv[1:] if argv is None else argv))
    logger.debug("Python %s, %s; рабочая папка «%s»", platform.python_version(), platform.platform(), Path.cwd())
    try:
        import numpy
        import pandas
        logger.debug("pandas %s, numpy %s", pandas.__version__, numpy.__version__)
    except ImportError:
        pass
    return path


def _remove_old(folder: Path, command: str, keep: int) -> None:
    logs = sorted(folder.glob(f"{command}_*.log"))
    for old in logs[:-keep] if keep > 0 else []:
        try:
            old.unlink()
        except OSError:
            pass


def explain(error: BaseException) -> tuple[str, str | None]:
    """Что не так и что сделать. Подсказка None — она уже есть в самом сообщении."""
    from tns.data.validator import ValidationError  # здесь, чтобы tns.logs не тянул pandas при импорте

    name = getattr(error, "filename", None)
    if isinstance(error, ValidationError):
        return str(error), None
    if isinstance(error, yaml.YAMLError):
        mark = getattr(error, "problem_mark", None)
        where = f", строка {mark.line + 1}" if mark is not None else ""
        return (f"файл настроек не читается{where}: {getattr(error, 'problem', None) or error}",
                "Исправьте эту строку в configs/settings.yaml (отступы — пробелами, после имени параметра двоеточие) "
                "или верните файл из репозитория.")
    if isinstance(error, PermissionError):
        return (f"нет доступа к «{name or error}»",
                "Если файл открыт в Excel или другой программе, закройте его. Проверьте, что у вас есть права "
                "на запись в эту папку, и запустите снова.")
    if isinstance(error, OSError) and error.errno == errno.ENOSPC:
        return "на диске закончилось место", "Освободите место на диске (или укажите другую папку в --out) и запустите снова."
    if isinstance(error, MemoryError):
        return ("не хватает оперативной памяти",
                "Закройте другие программы и запустите снова. Если не помогло — запустите на компьютере с большим объёмом памяти.")
    if isinstance(error, ModuleNotFoundError):
        return (f"не установлена библиотека «{error.name}»",
                f"Установите зависимости командой {REQUIREMENTS} и запустите снова.")
    if isinstance(error, KeyError):
        key = error.args[0] if error.args else error
        return (f"не найден параметр или колонка «{key}»",
                "Если вы меняли configs/settings.yaml, сравните его с версией из репозитория и верните недостающий параметр. "
                "Если не меняли — это ошибка программы: пришлите файл журнала разработчикам.")
    # Системные ошибки файлов (у них есть код errno). Свои сообщения программа пишет без кода — они ниже.
    if isinstance(error, OSError) and error.errno is not None:
        if isinstance(error, FileNotFoundError):
            return (f"не найден файл или папка «{name or error}»",
                    "Проверьте путь (параметры --data, --out, --config) и запустите снова.")
        if isinstance(error, (NotADirectoryError, FileExistsError)):
            return (f"по пути «{name or error}» лежит файл, а нужна папка",
                    "Укажите в --out папку (или переименуйте мешающий файл) и запустите снова.")
        return (f"не удалось прочитать или записать «{name or ''}»: {error.strerror or error}",
                "Проверьте, что путь доступен и файл не открыт в другой программе, и запустите снова.")
    if isinstance(error, (ValueError, OSError)):
        return str(error), None
    return (f"внутренняя ошибка программы ({type(error).__name__}: {error})",
            "Это не ошибка в ваших данных. Пришлите файл журнала разработчикам.")


def report_error(error: BaseException, log_path: Path | None, prefix: str = "") -> None:
    """Пишет ошибку: в консоль — что не так и что сделать, в журнал — ещё и трассировку."""
    if isinstance(error, KeyboardInterrupt):
        logger.error("Запуск прерван (Ctrl+C). Результаты последнего шага могли не записаться — запустите снова.")
        return
    message, hint = explain(error)
    lines = [f"{prefix}{message}" if prefix else message[:1].upper() + message[1:]]
    if hint:
        lines.append(f"Что сделать: {hint}")
    if log_path is not None:
        lines.append(f"Подробности — в журнале «{log_path}».")
    logger.error("\n".join(lines))
    logger.debug("Трассировка ошибки:", exc_info=(type(error), error, error.__traceback__))
