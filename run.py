#!/usr/bin/env python3
"""Единая точка запуска: план мер воздействия на месяц одной командой.

    python run.py --month 2026-02 --data "Материалы к кейсу"

Параметры — python run.py --help. Шаги описаны в tns/pipeline.py.
"""
import sys

try:
    from tns.pipeline import main
except ModuleNotFoundError as error:
    # Библиотеки ещё не загружены, поэтому журнала нет: пишем прямо на экран.
    print(
        f"ОШИБКА Не установлена библиотека «{error.name}».\n"
        "Что сделать: установите зависимости командой python -m pip install -r requirements.txt "
        "и запустите снова.",
        file=sys.stderr,
    )
    sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())
