#!/usr/bin/env python3
"""Единая точка запуска: план мер воздействия на месяц одной командой.

    python run.py --month 2026-02 --data "Материалы к кейсу"

Параметры — python run.py --help. Шаги описаны в tns/pipeline.py.
"""
import sys

from tns.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
