#!/usr/bin/env python3
"""
GUI для построения признаков K-Prototypes и получения рекомендаций
Использует PySide6
"""

import sys
import os
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QTextEdit,
    QGroupBox, QCheckBox, QSpinBox, QComboBox, QProgressBar,
    QMessageBox, QSplitter, QFrame
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QTextCursor, QIcon


class WorkerThread(QThread):
    """Поток для выполнения одной команды"""
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, command: list, working_dir: str):
        super().__init__()
        self.command = command
        self.working_dir = working_dir
        self.process = None

    def run(self):
        try:
            self.process = subprocess.Popen(
                self.command,
                cwd=self.working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
            for line in self.process.stdout:
                self.log_signal.emit(line.strip())
            return_code = self.process.wait()
            if return_code == 0:
                self.finished_signal.emit(True, "✅ Процесс завершён успешно!")
            else:
                self.finished_signal.emit(False, f"❌ Ошибка! Код возврата: {return_code}")
        except Exception as e:
            self.finished_signal.emit(False, f"❌ Ошибка выполнения: {str(e)}")

    def stop(self):
        if self.process:
            self.process.terminate()


class PipelineWorkerThread(QThread):
    """Поток для последовательного запуска нескольких скриптов (end-to-end pipeline)."""
    log_signal = Signal(str)
    finished_signal = Signal(bool, str)

    def __init__(self, commands: list, working_dir: str):
        super().__init__()
        self.commands = commands  # list of (label: str, cmd: list)
        self.working_dir = working_dir
        self.process = None
        self._stopped = False

    def run(self):
        for i, (label, cmd) in enumerate(self.commands, 1):
            if self._stopped:
                self.finished_signal.emit(False, "Прервано пользователем")
                return
            self.log_signal.emit(f"\n{'=' * 60}")
            self.log_signal.emit(f"[{i}/{len(self.commands)}] {label}")
            self.log_signal.emit(f"Команда: {' '.join(cmd)}")
            self.log_signal.emit("=" * 60)
            try:
                self.process = subprocess.Popen(
                    cmd,
                    cwd=self.working_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )
                for line in self.process.stdout:
                    self.log_signal.emit(line.strip())
                return_code = self.process.wait()
                if return_code != 0:
                    self.finished_signal.emit(
                        False, f"❌ Шаг {i} ({label}) завершился с ошибкой (код {return_code})"
                    )
                    return
                self.log_signal.emit(f"✅ Шаг {i} завершён успешно")
            except Exception as e:
                self.finished_signal.emit(False, f"❌ Ошибка на шаге {i} ({label}): {e}")
                return
        self.finished_signal.emit(True, "✅ Весь пайплайн завершён успешно!")

    def stop(self):
        self._stopped = True
        if self.process:
            self.process.terminate()


class RecommendationSystemGUI(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.current_data_dir = None
        self.init_ui()
        
    def init_ui(self):
        """Инициализация интерфейса"""
        self.setWindowTitle("Система рекомендаций по работе с должниками")
        self.setMinimumSize(1200, 800)
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Основной layout
        main_layout = QVBoxLayout(central_widget)
        
        # Создаём разделитель для левой и правой панели
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # ========== ЛЕВАЯ ПАНЕЛЬ (Параметры) ==========
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        
        # 1. Выбор директории с файлами
        dir_group = QGroupBox("1. Выбор исходных данных")
        dir_layout = QVBoxLayout(dir_group)
        
        dir_select_layout = QHBoxLayout()
        self.dir_path_edit = QLineEdit()
        self.dir_path_edit.setPlaceholderText("Путь к папке с файлами 01..13")
        self.dir_path_edit.setText("Материалы к кейсу")
        self.dir_browse_btn = QPushButton("Обзор...")
        self.dir_browse_btn.clicked.connect(self.browse_data_dir)
        
        dir_select_layout.addWidget(self.dir_path_edit)
        dir_select_layout.addWidget(self.dir_browse_btn)
        dir_layout.addLayout(dir_select_layout)
        
        # Кнопка проверки файлов
        self.check_files_btn = QPushButton("🔍 Проверить наличие файлов")
        self.check_files_btn.clicked.connect(self.check_files)
        dir_layout.addWidget(self.check_files_btn)
        
        left_layout.addWidget(dir_group)
        
        # 2. Параметры даты
        date_group = QGroupBox("2. Параметры даты")
        date_layout = QVBoxLayout(date_group)

        date_layout.addWidget(QLabel("Дата снимка признаков (опционально):"))
        self.snapshot_month_edit = QLineEdit()
        self.snapshot_month_edit.setPlaceholderText("YYYY-MM-01 (например, 2026-02-01)")
        self.snapshot_month_edit.setToolTip("Если не указано, используется максимальная дата из ОСВ")
        date_layout.addWidget(self.snapshot_month_edit)

        date_layout.addWidget(QLabel("Месяц рекомендаций (YYYY-MM):"))
        self.rec_month_edit = QLineEdit()
        self.rec_month_edit.setPlaceholderText("2026-02 (по умолчанию — текущий месяц)")
        self.rec_month_edit.setToolTip("Месяц, на который формируется план мер")
        date_layout.addWidget(self.rec_month_edit)

        left_layout.addWidget(date_group)
        
        # 3. Фильтры
        filter_group = QGroupBox("3. Фильтрация данных")
        filter_layout = QVBoxLayout(filter_group)
        
        self.only_delinquent_ckb = QCheckBox("Только должники (DZ_current > 0)")
        self.only_delinquent_ckb.setToolTip("Оставляет только ЛС с положительной задолженностью")
        filter_layout.addWidget(self.only_delinquent_ckb)
        
        left_layout.addWidget(filter_group)
        
        # 4. Параметры вывода
        output_group = QGroupBox("4. Пути для сохранения")
        output_layout = QVBoxLayout(output_group)
        
        output_layout.addWidget(QLabel("CSV файл:"))
        self.output_csv_edit = QLineEdit()
        self.output_csv_edit.setText("output/kprototypes_features.csv")
        output_layout.addWidget(self.output_csv_edit)
        
        output_layout.addWidget(QLabel("Parquet файл:"))
        self.output_parquet_edit = QLineEdit()
        self.output_parquet_edit.setText("output/kprototypes_features.parquet")
        output_layout.addWidget(self.output_parquet_edit)
        
        left_layout.addWidget(output_group)
        
        # 5. Кнопка запуска
        run_group = QGroupBox("5. Запуск")
        run_layout = QVBoxLayout(run_group)

        # Быстрый запуск — только признаки
        self.run_btn = QPushButton("📊 Построить признаки (шаг 1)")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 13px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.run_btn.clicked.connect(self.run_recommendations)
        run_layout.addWidget(self.run_btn)

        # End-to-end пайплайн
        self.pipeline_btn = QPushButton("🚀 СФОРМИРОВАТЬ ПЛАН МЕР НА МЕСЯЦ (end-to-end)")
        self.pipeline_btn.setMinimumHeight(50)
        self.pipeline_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #388E3C; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.pipeline_btn.clicked.connect(self.run_full_pipeline)
        run_layout.addWidget(self.pipeline_btn)

        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        run_layout.addWidget(self.progress_bar)

        left_layout.addWidget(run_group)
        
        # Кнопка открытия выходной папки
        self.open_folder_btn = QPushButton("📁 Открыть папку с результатами")
        self.open_folder_btn.clicked.connect(self.open_output_folder)
        self.open_folder_btn.setEnabled(False)
        left_layout.addWidget(self.open_folder_btn)
        
        left_layout.addStretch()
        
        # ========== ПРАВАЯ ПАНЕЛЬ (Лог) ==========
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        log_group = QGroupBox("Лог выполнения")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Menlo", 10))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border-radius: 5px;
            }
        """)
        log_layout.addWidget(self.log_text)
        
        # Кнопки управления логом
        log_buttons_layout = QHBoxLayout()
        self.clear_log_btn = QPushButton("🧹 Очистить лог")
        self.clear_log_btn.clicked.connect(self.clear_log)
        self.save_log_btn = QPushButton("💾 Сохранить лог")
        self.save_log_btn.clicked.connect(self.save_log)
        log_buttons_layout.addWidget(self.clear_log_btn)
        log_buttons_layout.addWidget(self.save_log_btn)
        log_layout.addLayout(log_buttons_layout)
        
        right_layout.addWidget(log_group)
        
        # Добавляем панели в сплиттер
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])
        
        # Статус бар
        self.statusBar().showMessage("Готов к работе")
        
        # Добавляем приветственное сообщение
        self.add_log("=" * 70)
        self.add_log("Система рекомендаций по работе с должниками")
        self.add_log("=" * 70)
        self.add_log("Настройте параметры в левой панели и нажмите 'Получить рекомендации'")
        self.add_log("")
        
    def add_log(self, message: str):
        """Добавление сообщения в лог"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # Прокрутка вниз
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
        
    def clear_log(self):
        """Очистка лога"""
        self.log_text.clear()
        self.add_log("Лог очищен")
        
    def save_log(self):
        """Сохранение лога в файл"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить лог",
            f"logs/recommendations_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.log_text.toPlainText())
            QMessageBox.information(self, "Успех", f"Лог сохранён в:\n{file_path}")
            
    def browse_data_dir(self):
        """Выбор директории с данными"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку с файлами 01..13",
            self.dir_path_edit.text() or os.path.expanduser("~")
        )
        if dir_path:
            self.dir_path_edit.setText(dir_path)
            self.current_data_dir = dir_path
            self.add_log(f"📁 Выбрана директория: {dir_path}")
            
    def check_files(self):
        """Проверка наличия необходимых файлов"""
        data_dir = Path(self.dir_path_edit.text())
        
        if not data_dir.exists():
            self.add_log(f"❌ Директория не существует: {data_dir}")
            return
            
        expected_prefixes = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']
        found_files = []
        missing_files = []
        
        for prefix in expected_prefixes:
            files = list(data_dir.glob(f"{prefix}*"))
            if files:
                found_files.append(files[0].name)
            else:
                missing_files.append(f"{prefix}*")
                
        self.add_log(f"\n📂 Проверка директории: {data_dir}")
        self.add_log(f"Найдено файлов: {len(found_files)}/13")
        
        if found_files:
            self.add_log("Найденные файлы:")
            for f in found_files:
                self.add_log(f"  ✅ {f}")
                
        if missing_files:
            self.add_log("Отсутствуют файлы:")
            for m in missing_files:
                self.add_log(f"  ❌ {m}")
                
        if len(found_files) >= 10:
            self.add_log("✅ Достаточно файлов для работы (минимум 10)")
        else:
            self.add_log("⚠️ Недостаточно файлов, проверьте директорию")
            
    def _scripts_dir(self) -> Path:
        """Возвращает путь к папке scripts/ рядом с interface.py."""
        return Path(__file__).parent / "scripts"

    def build_command(self) -> list:
        """Команда для шага 1: построение признаков."""
        script_path = self._scripts_dir() / "build_kprototypes_features.py"
        if not script_path.exists():
            self.add_log(f"❌ Не найден скрипт: {script_path}")
            return []

        cmd = [sys.executable, str(script_path)]
        cmd.extend(["--data-dir", str(Path(self.dir_path_edit.text()))])

        if self.snapshot_month_edit.text().strip():
            cmd.extend(["--snapshot-month", self.snapshot_month_edit.text().strip()])
        if self.output_csv_edit.text().strip():
            cmd.extend(["--output-csv", self.output_csv_edit.text().strip()])
        if self.output_parquet_edit.text().strip():
            cmd.extend(["--output-parquet", self.output_parquet_edit.text().strip()])
        if self.only_delinquent_ckb.isChecked():
            cmd.append("--only-delinquent")

        return cmd

    def build_pipeline_commands(self) -> list:
        """Собирает список команд для полного пайплайна (4 шага)."""
        scripts = self._scripts_dir()
        data_dir = str(Path(self.dir_path_edit.text()))
        month = self.rec_month_edit.text().strip() or None

        steps = []

        # Шаг 1: признаки
        feat_script = scripts / "build_kprototypes_features.py"
        if not feat_script.exists():
            self.add_log(f"❌ Не найден скрипт: {feat_script}")
            return []
        cmd1 = [sys.executable, str(feat_script), "--data-dir", data_dir]
        if self.snapshot_month_edit.text().strip():
            cmd1 += ["--snapshot-month", self.snapshot_month_edit.text().strip()]
        if self.output_csv_edit.text().strip():
            cmd1 += ["--output-csv", self.output_csv_edit.text().strip()]
        if self.output_parquet_edit.text().strip():
            cmd1 += ["--output-parquet", self.output_parquet_edit.text().strip()]
        if self.only_delinquent_ckb.isChecked():
            cmd1.append("--only-delinquent")
        steps.append(("Построение признаков (build_kprototypes_features)", cmd1))

        # Шаг 2: сегментация
        seg_script = scripts / "run_kprototypes_segmentation.py"
        if seg_script.exists():
            steps.append(("Сегментация K-Prototypes (run_kprototypes_segmentation)", [
                sys.executable, str(seg_script),
            ]))
        else:
            self.add_log(f"⚠️ Не найден скрипт: {seg_script} — шаг 2 пропущен")

        # Шаг 3: uplift/response модели
        resp_script = scripts / "build_response_models.py"
        if resp_script.exists():
            steps.append(("Uplift-модели (build_response_models)", [
                sys.executable, str(resp_script), "--data-dir", data_dir,
            ]))
        else:
            self.add_log(f"⚠️ Не найден скрипт: {resp_script} — шаг 3 пропущен")

        # Шаг 4: рекомендации
        rec_script = scripts / "build_recommendations.py"
        if not rec_script.exists():
            self.add_log(f"❌ Не найден скрипт: {rec_script}")
            return []
        cmd4 = [sys.executable, str(rec_script), "--data-dir", data_dir]
        if month:
            cmd4 += ["--month", month]
        steps.append(("Формирование плана мер (build_recommendations)", cmd4))

        return steps
        
    def _set_buttons_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.pipeline_btn.setEnabled(not running)
        if running:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setVisible(False)
            self.run_btn.setText("📊 Построить признаки (шаг 1)")
            self.pipeline_btn.setText("🚀 СФОРМИРОВАТЬ ПЛАН МЕР НА МЕСЯЦ (end-to-end)")

    def run_recommendations(self):
        """Шаг 1: только построение признаков."""
        data_dir = Path(self.dir_path_edit.text())
        if not data_dir.exists():
            QMessageBox.warning(self, "Ошибка", f"Директория не существует:\n{data_dir}")
            return

        cmd = self.build_command()
        if not cmd:
            return

        self._set_buttons_running(True)
        self.run_btn.setText("⏳ Построение признаков...")

        self.add_log("\n" + "=" * 70)
        self.add_log("📊 ПОСТРОЕНИЕ ПРИЗНАКОВ")
        self.add_log("=" * 70)
        self.add_log(f"Команда: {' '.join(cmd)}")
        self.add_log("")

        self.worker = WorkerThread(cmd, str(Path(__file__).parent))
        self.worker.log_signal.connect(self.add_log)
        self.worker.finished_signal.connect(self.on_process_finished)
        self.worker.start()

    def run_full_pipeline(self):
        """End-to-end: признаки → сегментация → uplift-модели → рекомендации."""
        data_dir = Path(self.dir_path_edit.text())
        if not data_dir.exists():
            QMessageBox.warning(self, "Ошибка", f"Директория не существует:\n{data_dir}")
            return

        commands = self.build_pipeline_commands()
        if not commands:
            return

        self._set_buttons_running(True)
        self.pipeline_btn.setText("⏳ Пайплайн выполняется...")

        month = self.rec_month_edit.text().strip() or "текущий"
        self.add_log("\n" + "=" * 70)
        self.add_log(f"🚀 ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА (месяц: {month})")
        self.add_log(f"Шагов: {len(commands)}")
        self.add_log("=" * 70 + "\n")

        self.worker = PipelineWorkerThread(commands, str(Path(__file__).parent))
        self.worker.log_signal.connect(self.add_log)
        self.worker.finished_signal.connect(self.on_process_finished)
        self.worker.start()
        
    def on_process_finished(self, success: bool, message: str):
        """Обработка завершения процесса/пайплайна."""
        self._set_buttons_running(False)

        self.add_log("")
        self.add_log("-" * 70)
        self.add_log(message)
        self.add_log("-" * 70)

        if success:
            self.add_log("✨ Процесс завершён успешно!")

            output_dir = Path(self.output_csv_edit.text()).parent
            self.add_log(f"\n📄 Результаты в папке: {output_dir}")
            for f in sorted(output_dir.glob("recommendations*.csv")):
                size_mb = f.stat().st_size / (1024 * 1024)
                self.add_log(f"   {f.name}  ({size_mb:.1f} MB)")

            self.open_folder_btn.setEnabled(True)

            QMessageBox.information(
                self,
                "Успех",
                f"✅ Готово!\n\nРезультаты сохранены в папке:\n{output_dir}\n\n"
                f"Файлы: recommendations_*.csv, recommendations_summary.csv"
            )
            self.statusBar().showMessage("✅ Процесс завершён успешно")
        else:
            self.add_log("❌ Процесс завершился с ошибкой")
            QMessageBox.critical(
                self,
                "Ошибка",
                f"❌ Ошибка при выполнении!\n\n{message}\n\nПроверьте лог для деталей."
            )
            self.statusBar().showMessage("❌ Ошибка выполнения")
            
    def open_output_folder(self):
        """Открытие папки с выходными файлами"""
        output_dir = Path(self.output_csv_edit.text()).parent
        if output_dir.exists():
            # Для MacOS
            if sys.platform == 'darwin':
                subprocess.run(['open', str(output_dir)])
            # Для Windows
            elif sys.platform == 'win32':
                os.startfile(str(output_dir))
            # Для Linux
            else:
                subprocess.run(['xdg-open', str(output_dir)])
        else:
            QMessageBox.warning(self, "Ошибка", f"Папка не существует:\n{output_dir}")
            
    def closeEvent(self, event):
        """Обработка закрытия окна."""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Подтверждение",
                "Процесс ещё выполняется. Прервать и выйти?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.worker.stop()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = RecommendationSystemGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()