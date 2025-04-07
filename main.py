# import logging
# import numpy as np
# import random  # Import random для генерации координат
# from dmr_subscriber import DMRSubscriber
# from communication_modes import DirectMode, RepeaterMode
# from channel_model import AWGNChannel, FadingChannel
# from tdma import TDMAFrame, Repeater
# from visualization import DMRVisualizer
# from simulation import MODE_SIMULATION_MAP, direct_mode_simulation, repeater_mode_simulation
# from strategy import DirectModeSimulationStrategy, RepeaterModeSimulationStrategy
#
# from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
#                              QLabel, QPushButton, QListWidget, QListWidgetItem,
#                              QGridLayout, QSizePolicy)
# from PyQt5.QtCore import Qt, QTimer
# from PyQt5.QtGui import QColor, QBrush
# import sys
# import pyqtgraph as pg  # Import PyQtGraph
#
# import sys
# import random
# from PyQt5.QtWidgets import QApplication
# from PyQt5.QtCore import QTimer
#
# from dmr_subscriber import DMRSubscriber
# from strategy import DirectModeSimulationStrategy
# from gui import MainWindow # Импортируем класс MainWindow из gui.py
# from signal_processing import calculate_spectrum, generate_dmr_signal
#
# def generate_spectrum_data():
#     """Генерирует случайные данные для спектра (заглушка)."""
#     return [random.random() for _ in range(50)]
#
# # Настройка логгера
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#
#     #  Создаем абонентов
#     simulation_strategy = DirectModeSimulationStrategy() #  Пример стратегии
#     num_bits = 200
#     CARRIER_FREQ = 4000
#
#     subscribers = [
#         DMRSubscriber(id=1, latitude=random.uniform(-90, 90), longitude=random.uniform(-180, 180), simulation_strategy=simulation_strategy, carrier_frequency=CARRIER_FREQ + 100),
#         DMRSubscriber(id=2, latitude=random.uniform(-90, 90), longitude=random.uniform(-180, 180), simulation_strategy=simulation_strategy, carrier_frequency=CARRIER_FREQ - 100),
#     ]
#
#     # Создаем и отображаем окно
#     SAMPLE_RATE = 48000  #  Sample Rate, теперь это константа
#     window = MainWindow(subscribers, SAMPLE_RATE) # Передаем список абонентов в окно
#     window.show()
#
    # #  Таймер для периодической передачи сообщений абонентами
    # def transmit_messages():
    #     print("transmit_messages")
    #     for sub in subscribers:
    #         print(f'sub {sub.id}')
    #         sub.generate_random_message(num_bits) # Генерируем случайное сообщение
    #         sub.transmit()  #  Помещаем сигнал в буфер
    #
    # transmit_timer = QTimer()
    # transmit_timer.timeout.connect(transmit_messages)
    # transmit_timer.start(50) #  Каждые полсекунды
#
#     #  Таймер для периодического обновления данных спектра
#     def update_spectrum():
#         # 1. Суммируем сигналы от всех абонентов
#         combined_signal = np.zeros(int(0.1 * SAMPLE_RATE))  # 100ms
#         num_samples = len(combined_signal)  # Вычисляем правильный размер
#         print(num_samples)
#         #  Собираем частоты
#         subscriber_frequencies = []
#
#         for sub in subscribers:
#             signal = generate_dmr_signal(sub, num_samples, SAMPLE_RATE)
#             print(f"signal : {len(signal)}")
#             print(f"combined_signal : {len(combined_signal)}")
#             combined_signal += signal
#             #  Получаем текущие частоты
#             subscriber_frequencies.extend(sub.get_current_frequencies())
#
#         # 2. Вычисляем спектр
#         frequencies, spectrum = calculate_spectrum(combined_signal, SAMPLE_RATE)
#         frequencies = frequencies / 1e6
#
#         if frequencies is not None and spectrum is not None:
#             # 3. Обновляем график
#             window.update_spectrum(spectrum, frequencies, subscriber_frequencies)  # Передаем и spectrum, и frequencies
#         else:
#             print("Ошибка при вычислении спектра.")
#
#     timer = QTimer()
#     timer.timeout.connect(update_spectrum)      # Привязываем к функции обновления
#     timer.start(100)                            # Запускаем таймер (каждые 100 мс)
#
#     sys.exit(app.exec_())

# main.py
import sys
import random
import logging

import numpy as np
from PyQt5.QtWidgets import QApplication
from gui import MainWindow

# Настройка логгера
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    #  Создаем абонентов
    SAMPLE_RATE = 48000  # Sample Rate, теперь это константа
    CARRIER_FREQ = 4000
    num_bits = 20

    subscribers = [] #  Первоначальный список абонентов пуст, он будет создан SubscriberManager
    #  Таймер для периодической передачи сообщений абонентами
    window = MainWindow(subscribers, SAMPLE_RATE) #  Передаем ПУСТОЙ список абонентов
    window.setGeometry(100, 100, 1200, 800)
    window.show()

    sys.exit(app.exec_())