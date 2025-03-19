# import numpy as np
# import pyqtgraph as pg
# from pyqtgraph.Qt import QtCore, QtWidgets
# from scipy.signal import find_peaks
# from enum import Enum
# from collections import namedtuple
# from scipy import signal

# class SignalType(Enum):
#     DMR = 1
#     TETRA = 2
#     GSM = 3
#     CUSTOM = 4
#
#
# SignalConfig = namedtuple('SignalConfig', ['modulation', 'symbol_rate', 'color', 'marker'])
#
# # Конфигурация сигналов (легко расширяется)
# SIGNAL_CONFIGS = {
#     SignalType.DMR: SignalConfig(
#         modulation='4-FSK',
#         symbol_rate=4800,
#         color='#FF0000',
#         marker='s'
#     ),
#     SignalType.TETRA: SignalConfig(
#         modulation='π/4-DQPSK',
#         symbol_rate=18000,
#         color='#2ca02c',
#         marker='t'
#     ),
#     SignalType.GSM: SignalConfig(
#         modulation='GMSK',
#         symbol_rate=270833,
#         color='#d62728',
#         marker='o'
#     )
# }
#
# class AdvancedDOASystem:
#     def __init__(self):
#         # Параметры системы
#         self.num_antennas = 8
#         self.freq = 900e6
#         self.wavelength = 3e8 / self.freq
#         self.snr_db = 200
#         self.num_sources = 3
#         self.current_view = "cartesian"
#         self.max_spectrum = 1e6
#         self.range_limit = 1000  # Максимальная дальность на карте (метры)]
#         self.sampling_rate = 1e6
#         self.center_freq = 900e6
#
#         # Инициализация GUI
#         self.app = QtWidgets.QApplication([])
#         self.win = QtWidgets.QWidget()
#         self.layout = QtWidgets.QHBoxLayout(self.win)  # Горизонтальное разделение
#
#         # Правая панель: карта
#         self.left_panel = QtWidgets.QVBoxLayout()
#         self._init_map()
#         self.layout.addLayout(self.left_panel)
#
#         # Левая панель: графики
#         self.right_panel = QtWidgets.QVBoxLayout()
#         self._init_graphs()
#         self.layout.addLayout(self.right_panel)
#
#         # Позиции антенн
#         self.antenna_pos = self.create_antenna_array()
#         self.timer = QtCore.QTimer()
#         self.timer.timeout.connect(self.update)
#         self.timer.start(100)
#
#         # Инициализация абонентов
#         self.subscribers = [
#             self.create_subscriber(SignalType.DMR),
#             # self.create_subscriber(SignalType.TETRA),
#             # self.create_subscriber(SignalType.GSM)
#         ]
#
#         # Добавляем график спектра
#         self.init_spectrum_plot()
#
#     def _init_graphs(self):
#         # Панель управления графиками
#         self.toolbar = QtWidgets.QHBoxLayout()
#         self.switch_btn = QtWidgets.QPushButton("Переключить вид")
#         self.switch_btn.clicked.connect(self.toggle_view)
#         self.toolbar.addWidget(self.switch_btn)
#         self.right_panel.addLayout(self.toolbar)
#
#         # Контейнер для графиков
#         self.plot_stack = QtWidgets.QStackedWidget()
#         self.right_panel.addWidget(self.plot_stack)
#
#         # Декартов график
#         self.cartesian_plot = pg.PlotWidget(title="Декартовы координаты")
#         self.cartesian_curve = self.cartesian_plot.plot(pen='y')
#         self.cartesian_markers = [
#             pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(color, width=2))
#             for color in ['#FF0000', '#00FF00', '#0000FF']
#         ]
#         for m in self.cartesian_markers:
#             self.cartesian_plot.addItem(m)
#         self.plot_stack.addWidget(self.cartesian_plot)
#
#         # Полярный график
#         self.polar_plot = pg.PlotWidget(title="Полярные координаты")
#         self._init_polar_plot()
#         self.polar_curve = self.polar_plot.plot(pen=pg.mkPen('#00FF00', width=2))
#         self.polar_markers = [
#             pg.ScatterPlotItem(symbol='o', size=10, pen=pg.mkPen(color, width=2))
#             for color in ['#FF0000', '#00FFFF', '#FF00FF']
#         ]
#         for m in self.polar_markers:
#             self.polar_plot.addItem(m)
#         self.plot_stack.addWidget(self.polar_plot)
#
#     def _init_map(self):
#         """Инициалиазация карты местности"""
#         self.map_plot = pg.PlotWidget(title="Карта местности")
#         self.map_plot.setAspectLocked(True)
#         self.map_plot.setXRange(-self.range_limit, self.range_limit)
#         self.map_plot.setYRange(-self.range_limit, self.range_limit)
#
#         # Настройка осей
#         self.map_plot.setLabel('left', 'Y, м')
#         self.map_plot.setLabel('bottom', 'X, м')
#         grid_pen = pg.mkPen('#808080', width=1, style=QtCore.Qt.PenStyle.DashLine)
#
#         # Вертикальные линии (X)
#         for x in np.arange(-self.range_limit, self.range_limit + 100, 100):
#             line = pg.InfiniteLine(
#                 pos=x,
#                 angle=90,
#                 pen=grid_pen,
#                 movable=False
#             )
#             self.map_plot.addItem(line)
#
#         # Горизонтальные линии (Y)
#         for y in np.arange(-self.range_limit, self.range_limit + 100, 100):
#             line = pg.InfiniteLine(
#                 pos=y,
#                 angle=0,
#                 pen=grid_pen,
#                 movable=False
#             )
#             self.map_plot.addItem(line)
#
#         # Основные оси
#         x_zero = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('#000000', width=1))
#         y_zero = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('#000000', width=1))
#         self.map_plot.addItem(x_zero)
#         self.map_plot.addItem(y_zero)
#
#         # Позиция пеленгатора (центр)
#         self.doa_marker = pg.ScatterPlotItem(
#             pos=[(0, 0)],
#             symbol='s',
#             size=15,
#             pen=pg.mkPen('#000000', width=2),
#             brush=pg.mkBrush('#FFFF00')
#         )
#         self.map_plot.addItem(self.doa_marker)
#
#         # Направления на абонентов
#         self.target_lines = [
#             pg.PlotCurveItem(pen=pg.mkPen(color, width=2))
#             for color in ['#FF0000', '#00FF00', '#0000FF']
#         ]
#         for line in self.target_lines:
#             self.map_plot.addItem(line)
#
#         # Маркеры позиций абонентов
#         self.target_markers = [
#             pg.ScatterPlotItem(
#                 symbol='o',
#                 size=12,
#                 pen=pg.mkPen(color, width=1.5),
#                 brush=pg.mkBrush(color)
#             )
#             for color in ['#FF0000', '#00FF00', '#0000FF']
#         ]
#         for marker in self.target_markers:
#             self.map_plot.addItem(marker)
#
#         # Текст с номерами абонентов
#         self.target_labels = [
#             pg.TextItem(text=f"A{i + 1}", color=color, anchor=(0.5, 0))
#             for i, color in enumerate(['#FF0000', '#00FF00', '#0000FF'])
#         ]
#         for label in self.target_labels:
#             self.map_plot.addItem(label)
#
#         self.left_panel.addWidget(self.map_plot)
#
#     def _init_polar_plot(self):
#         self.polar_plot.setAspectLocked(True)
#         self.polar_plot.hideAxis('left')
#         self.polar_plot.hideAxis('bottom')
#         for angle in np.arange(0, 360, 30):
#             x = [0, 1.1 * np.cos(np.radians(angle))]
#             y = [0, 1.1 * np.sin(np.radians(angle))]
#             self.polar_plot.addItem(pg.PlotCurveItem(x, y, pen=pg.mkPen('#404040', width=0.5)))
#         for r in np.linspace(0.25, 1.0, 4):
#             theta = np.linspace(0, 2 * np.pi, 100)
#             x = r * np.cos(theta)
#             y = r * np.sin(theta)
#             self.polar_plot.addItem(pg.PlotCurveItem(x, y, pen=pg.mkPen('#404040', width=0.5)))
#
#     def init_spectrum_plot(self):
#         """Инициализация графика спектра"""
#         self.spectrum_plot = pg.PlotWidget(title="Спектральное полотно")
#         self.spectrum_plot.setLabel('left', 'Уровень, дБ')
#         self.spectrum_plot.setLabel('bottom', 'Частота, МГц')
#         self.spectrum_curve = self.spectrum_plot.plot(pen='#1f77b4')
#
#         # Добавляем в layout
#         self.left_panel.addWidget(self.spectrum_plot)
#
#     def calculate_spectrum(self, signals):
#         """Вычисление спектра сигналов"""
#         # Применяем оконную функцию
#         window = np.hanning(signals.shape[1])
#         windowed = signals * window[np.newaxis, :]
#
#         # Вычисляем FFT
#         fft = np.fft.fft(windowed, axis=1)
#         fft_shifted = np.fft.fftshift(fft, axes=1)
#
#         # Преобразуем в дБ
#         power = np.abs(fft_shifted) ** 2
#         db = 10 * np.log10(power / np.max(power))
#
#         # Частотная ось
#         freq = np.fft.fftshift(np.fft.fftfreq(
#             signals.shape[1],
#             1 / self.sampling_rate
#         )) + self.center_freq
#
#         return freq / 1e6, db.mean(axis=0)  # Усредняем по антеннам
#
#     def toggle_view(self):
#         self.current_view = "polar" if self.current_view == "cartesian" else "cartesian"
#         self.plot_stack.setCurrentIndex(0 if self.current_view == "cartesian" else 1)
#
#     def create_antenna_array(self):
#         angles = np.linspace(0, 2 * np.pi, self.num_antennas, endpoint=False)
#         return np.column_stack([np.cos(angles), np.sin(angles)]) * (self.wavelength / 2)
#
#     # Генерация сигналов
#     @staticmethod
#     def generate_dmr_signal(num_samples):
#         bits = np.random.randint(0, 4, num_samples)
#         t = np.linspace(0, num_samples / 4800, num_samples)
#         return np.cos(2 * np.pi * 0.5 * 4800 * t + np.pi / 2 * bits)
#
#     def generate_signals(self):
#         """Генерация сигналов для всех антенн"""
#         num_samples = 1024  # Количество временных отсчетов
#         signals = np.zeros((self.num_antennas, num_samples), dtype=np.complex64)
#
#         for sub in self.subscribers:
#             # Генерация сигнала источника
#             if sub['type'] == SignalType.DMR:
#                 source_signal = self.generate_dmr_signal(num_samples)
#             # ... другие типы сигналов ...
#
#             # Расчет задержки для антенной решетки
#             delay_phase = np.exp(-2j * np.pi * self.freq *
#                                  (self.antenna_pos[:, 0] * np.cos(np.radians(sub['angle'])) +
#                                   self.antenna_pos[:, 1] * np.sin(np.radians(sub['angle'])) / 3e8))
#
#             # Добавление сигнала с учетом задержки и затухания
#             signals += (source_signal * delay_phase[:, np.newaxis] *
#                         0.9 ** (sub['distance'] / 100))
#
#             # Добавление шума
#             noise = 10 ** (-self.snr_db / 20) * (np.random.randn(*signals.shape) +
#                                                  1j * np.random.randn(*signals.shape))
#         return signals + noise
#
#     def create_subscriber(self, signal_type):
#         """Создает абонента с заданным типом сигнала"""
#         return {
#             'type': signal_type,
#             'distance': np.random.uniform(100, self.range_limit),
#             'angle': np.random.uniform(0, 360),
#             'config': SIGNAL_CONFIGS[signal_type]
#         }
#
#     def music_algorithm(self, signals):
#         R = (signals @ signals.conj().T) / signals.shape[1]
#         eigvals, eigvecs = np.linalg.eigh(R)
#         noise_subspace = eigvecs[:, :-self.num_sources]
#
#         theta_range = np.linspace(0, 360, 360)
#         spectrum = np.zeros_like(theta_range)
#
#         for i, theta in enumerate(theta_range):
#             a = np.exp(-1j * 2 * np.pi * (
#                     self.antenna_pos[:, 0] * np.cos(np.radians(theta)) +
#                     self.antenna_pos[:, 1] * np.sin(np.radians(theta))
#             ) / self.wavelength)
#             spectrum[i] = 1 / np.linalg.norm(noise_subspace.conj().T @ a) ** 2
#
#         peaks, _ = find_peaks(spectrum, height=0.5 * np.max(spectrum), distance=10)
#         estimated_angles = theta_range[peaks][:self.num_sources]  # Ограничение по числу источников
#
#         return estimated_angles, theta_range, spectrum / self.max_spectrum
#
#     def update_map(self, estimated_angles):
#         """Обновление карты с типами сигналов"""
#         for i, sub in enumerate(self.subscribers):
#             # Реальные позиции
#             x_real = sub['distance'] * np.cos(np.radians(sub['angle']))
#             y_real = sub['distance'] * np.sin(np.radians(sub['angle']))
#
#             print("x_real:", x_real)
#             print("y_real:", y_real)
#
#             # Стиль маркера из конфига
#             config = sub['config']
#             self.target_markers[i].setData(
#                 [x_real],
#                 [y_real],
#                 symbol=config.marker,
#                 pen=config.color,
#                 brush=config.color
#             )
#
#             # Подписи с типом сигнала
#             label = self.target_labels[i]
#             label.setText(f"{sub['type'].name}\n{sub['distance']:.0f} м")
#             label.setPos(x_real, y_real)
#             # self.target_labels[i].setText(f"{sub['type'].name}\n{config.modulation}")
#
#         """Обновление карты: линии - вычисленные направления, маркеры - реальные позиции"""
#         display_distance = self.range_limit * 0.8  # 80% радиуса карты
#
#         # Очистка предыдущих направлений
#         for line in self.target_lines:
#             line.setData([], [])
#
#         print("Angle:", estimated_angles)
#         # Рисование новых направлений
#         for i, angle in enumerate(estimated_angles):
#             x = display_distance * np.cos(np.radians(angle))
#             y = display_distance * np.sin(np.radians(angle))
#             color = ['#FF0000', '#00FF00', '#0000FF'][i % 3]
#
#             self.target_lines[i].setData(
#                 [0, x],
#                 [0, y],
#                 pen=pg.mkPen(color, width=2, style=QtCore.Qt.PenStyle.DashLine)
#             )
#
#         """Обновление карты с реальными позициями абонентов"""
#         for i, sub in enumerate(self.subscribers):
#             x_real = sub['distance'] * np.cos(np.radians(sub['angle']))
#             y_real = sub['distance'] * np.sin(np.radians(sub['angle']))
#
#             # Обновление маркеров
#             self.target_markers[i].setData([x_real], [y_real])
#
#     def update(self):
#         """Основной цикл обновления данных"""
#         # 1. Генерация сигналов
#         signals = self.generate_signals()
#
#         # 2. Обработка алгоритмом MUSIC
#         estimated_angles, theta_range, spectrum = self.music_algorithm(signals)
#
#         print("Signals shape:", signals.shape)  # Должно быть (8, 1024)
#
#         # 3. Обновление графиков
#         self.update_plots(spectrum, estimated_angles)
#
#         # 4. Обновление карты
#         self.update_map(estimated_angles)
#
#         freq, spectrum = self.calculate_spectrum(signals)
#         self.spectrum_curve.setData(freq, spectrum)
#
#     def update_plots(self, spectrum, estimated_angles):
#         """Обновление графиков"""
#         # Декартов график
#         self.cartesian_curve.setData(
#             np.linspace(0, 360, len(spectrum)),
#             spectrum
#         )
#
#         # Полярный график
#         theta_rad = np.radians(np.linspace(0, 360, len(spectrum)))
#         x = spectrum * np.cos(theta_rad)
#         y = spectrum * np.sin(theta_rad)
#         self.polar_curve.setData(x, y)
#
#         # Маркеры направлений
#         for i, m in enumerate(self.cartesian_markers):
#             if i < len(estimated_angles):
#                 m.setPos(estimated_angles[i])
#                 m.setVisible(True)
#             else:
#                 m.setVisible(False)
#
#     def run(self):
#         self.win.show()
#         self.app.exec()
#
#
# if __name__ == "__main__":
#     system = AdvancedDOASystem()
#     system.run()

from PyQt6.QtWidgets import QApplication
from signal_processor import SignalProcessor
from visualizer import Visualizer
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.signal import find_peaks
from enum import Enum
from collections import namedtuple
from scipy import signal


class App(QApplication):
    def __init__(self):
        super().__init__([])
        self.signal_processor = SignalProcessor()
        self.visualizer = Visualizer()
        self.visualizer.show()

        # Связываем обновление данных
        self.visualizer.timer.timeout.connect(self.process_data)

    def process_data(self):
        self.signal_processor.generate_subscribers()
        signals = self.signal_processor.generate_signals()
        angles, spectrum = self.signal_processor.process_signals(signals)
        freqs, spectrum_db = self.signal_processor.calculate_spectrum(signals)
        self.visualizer.update_plots(angles, spectrum, freqs, spectrum_db, self.signal_processor.subscribers )

if __name__ == "__main__":
    app = App()
    app.exec()