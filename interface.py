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
    """Поток для выполнения долгих операций"""
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
                universal_newlines=True
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
        
        date_layout.addWidget(QLabel("Дата снимка (опционально):"))
        self.snapshot_month_edit = QLineEdit()
        self.snapshot_month_edit.setPlaceholderText("YYYY-MM-01 (например, 2026-02-01)")
        self.snapshot_month_edit.setToolTip("Если не указано, используется максимальная дата из ОСВ")
        date_layout.addWidget(self.snapshot_month_edit)
        
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
        
        self.run_btn = QPushButton("🚀 ПОЛУЧИТЬ РЕКОМЕНДАЦИИ")
        self.run_btn.setMinimumHeight(50)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.run_btn.clicked.connect(self.run_recommendations)
        run_layout.addWidget(self.run_btn)
        
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
            
    def build_command(self) -> list:
        """Сборка команды для запуска"""
        # Определяем путь к скрипту
        script_path = Path(__file__).parent / "build_kprototypes_features.py"
        if not script_path.exists():
            # Пробуем найти в текущей директории
            script_path = Path.cwd() / "build_kprototypes_features.py"
            
        if not script_path.exists():
            self.add_log("❌ Не найден скрипт build_kprototypes_features.py")
            return []
            
        cmd = [sys.executable, str(script_path)]
        
        # Добавляем параметры
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
        
    def run_recommendations(self):
        """Запуск процесса построения рекомендаций"""
        # Проверяем директорию
        data_dir = Path(self.dir_path_edit.text())
        if not data_dir.exists():
            QMessageBox.warning(self, "Ошибка", f"Директория не существует:\n{data_dir}")
            return
            
        # Собираем команду
        cmd = self.build_command()
        if not cmd:
            return
            
        # Блокируем кнопку
        self.run_btn.setEnabled(False)
        self.run_btn.setText("⏳ Обработка данных...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Бесконечный прогресс
        
        # Выводим команду в лог
        self.add_log("\n" + "=" * 70)
        self.add_log("🚀 ЗАПУСК ПРОЦЕССА ПОСТРОЕНИЯ РЕКОМЕНДАЦИЙ")
        self.add_log("=" * 70)
        self.add_log(f"Команда: {' '.join(cmd)}")
        self.add_log("")
        
        # Запускаем поток
        self.worker = WorkerThread(cmd, str(Path.cwd()))
        self.worker.log_signal.connect(self.add_log)
        self.worker.finished_signal.connect(self.on_process_finished)
        self.worker.start()
        
    def on_process_finished(self, success: bool, message: str):
        """Обработка завершения процесса"""
        self.run_btn.setEnabled(True)
        self.run_btn.setText("🚀 ПОЛУЧИТЬ РЕКОМЕНДАЦИИ")
        self.progress_bar.setVisible(False)
        
        self.add_log("")
        self.add_log("-" * 70)
        self.add_log(message)
        self.add_log("-" * 70)
        
        if success:
            self.add_log("✨ Рекомендации успешно сформированы!")
            
            # Показываем пути к выходным файлам
            output_csv = self.output_csv_edit.text()
            output_parquet = self.output_parquet_edit.text()
            
            self.add_log(f"\n📄 Выходные файлы:")
            self.add_log(f"   CSV: {output_csv}")
            self.add_log(f"   Parquet: {output_parquet}")
            
            # Проверяем существование файлов
            if Path(output_csv).exists():
                size_mb = Path(output_csv).stat().st_size / (1024 * 1024)
                self.add_log(f"   Размер CSV: {size_mb:.2f} MB")
                
            # Активируем кнопку открытия папки
            self.open_folder_btn.setEnabled(True)
            
            # Показываем уведомление
            QMessageBox.information(
                self,
                "Успех",
                f"✅ Рекомендации успешно сформированы!\n\n"
                f"Результаты сохранены в:\n"
                f"• {output_csv}\n"
                f"• {output_parquet}\n\n"
                f"Теперь можно запустить run_kprototypes_segmentation.py для кластеризации"
            )
            
            self.statusBar().showMessage("✅ Процесс завершён успешно")
        else:
            self.add_log("❌ Процесс завершился с ошибкой")
            QMessageBox.critical(
                self,
                "Ошибка",
                f"❌ Ошибка при выполнении!\n\n{message}\n\n"
                f"Проверьте лог для получения деталей."
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
        """Обработка закрытия окна"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Подтверждение",
                "Процесс ещё выполняется. Прервать и выйти?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
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