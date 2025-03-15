import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.signal import find_peaks
from scipy import signal

class AdvancedDOASystem:
    def __init__(self):
        # Параметры системы
        self.num_antennas = 8
        self.freq = 900e6
        self.wavelength = 3e8 / self.freq
        self.snr_db = 20
        self.num_sources = 3
        self.current_view = "cartesian"
        self.max_spectrum = 1e6
        self.range_limit = 1000  # Максимальная дальность на карте (метры)

        # Инициализация GUI
        self.app = QtWidgets.QApplication([])
        self.win = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout(self.win)  # Горизонтальное разделение

        # Правая панель: карта
        self.left_panel = QtWidgets.QVBoxLayout()
        self._init_map()
        self.layout.addLayout(self.left_panel)

        # Левая панель: графики
        self.right_panel = QtWidgets.QVBoxLayout()
        self._init_graphs()
        self.layout.addLayout(self.right_panel)

        # Позиции антенн
        self.antenna_pos = self.create_antenna_array()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(5000)

    def _init_graphs(self):
        # Панель управления графиками
        self.toolbar = QtWidgets.QHBoxLayout()
        self.switch_btn = QtWidgets.QPushButton("Переключить вид")
        self.switch_btn.clicked.connect(self.toggle_view)
        self.toolbar.addWidget(self.switch_btn)
        self.right_panel.addLayout(self.toolbar)

        # Контейнер для графиков
        self.plot_stack = QtWidgets.QStackedWidget()
        self.right_panel.addWidget(self.plot_stack)

        # Декартов график
        self.cartesian_plot = pg.PlotWidget(title="Декартовы координаты")
        self.cartesian_curve = self.cartesian_plot.plot(pen='y')
        self.cartesian_markers = [
            pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(color, width=2))
            for color in ['#FF0000', '#00FF00', '#0000FF']
        ]
        for m in self.cartesian_markers:
            self.cartesian_plot.addItem(m)
        self.plot_stack.addWidget(self.cartesian_plot)

        # Полярный график
        self.polar_plot = pg.PlotWidget(title="Полярные координаты")
        self._init_polar_plot()
        self.polar_curve = self.polar_plot.plot(pen=pg.mkPen('#00FF00', width=2))
        self.polar_markers = [
            pg.ScatterPlotItem(symbol='o', size=10, pen=pg.mkPen(color, width=2))
            for color in ['#FF0000', '#00FFFF', '#FF00FF']
        ]
        for m in self.polar_markers:
            self.polar_plot.addItem(m)
        self.plot_stack.addWidget(self.polar_plot)

    def _init_map(self):
        """Инициалиазация карты местности"""
        self.map_plot = pg.PlotWidget(title="Карта местности")
        self.map_plot.setAspectLocked(True)
        self.map_plot.setXRange(-self.range_limit, self.range_limit)
        self.map_plot.setYRange(-self.range_limit, self.range_limit)

        # Настройка осей
        self.map_plot.setLabel('left', 'Y, м')
        self.map_plot.setLabel('bottom', 'X, м')
        grid_pen = pg.mkPen('#808080', width=1, style=QtCore.Qt.PenStyle.DashLine)

        # Вертикальные линии (X)
        for x in np.arange(-self.range_limit, self.range_limit + 100, 100):
            line = pg.InfiniteLine(
                pos=x,
                angle=90,
                pen=grid_pen,
                movable=False
            )
            self.map_plot.addItem(line)

        # Горизонтальные линии (Y)
        for y in np.arange(-self.range_limit, self.range_limit + 100, 100):
            line = pg.InfiniteLine(
                pos=y,
                angle=0,
                pen=grid_pen,
                movable=False
            )
            self.map_plot.addItem(line)

        # Основные оси
        x_zero = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('#000000', width=1))
        y_zero = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('#000000', width=1))
        self.map_plot.addItem(x_zero)
        self.map_plot.addItem(y_zero)

        # Позиция пеленгатора (центр)
        self.doa_marker = pg.ScatterPlotItem(
            pos=[(0, 0)],
            symbol='s',
            size=15,
            pen=pg.mkPen('#000000', width=2),
            brush=pg.mkBrush('#FFFF00')
        )
        self.map_plot.addItem(self.doa_marker)

        # Направления на абонентов
        self.target_lines = [
            pg.PlotCurveItem(pen=pg.mkPen(color, width=2))
            for color in ['#FF0000', '#00FF00', '#0000FF']
        ]
        for line in self.target_lines:
            self.map_plot.addItem(line)

        # Маркеры позиций абонентов
        self.target_markers = [
            pg.ScatterPlotItem(
                symbol='o',
                size=12,
                pen=pg.mkPen(color, width=1.5),
                brush=pg.mkBrush(color)
            )
            for color in ['#FF0000', '#00FF00', '#0000FF']
        ]
        for marker in self.target_markers:
            self.map_plot.addItem(marker)

        # Текст с номерами абонентов
        self.target_labels = [
            pg.TextItem(text=f"A{i + 1}", color=color, anchor=(0.5, 1.5))
            for i, color in enumerate(['#FF0000', '#00FF00', '#0000FF'])
        ]
        for label in self.target_labels:
            self.map_plot.addItem(label)

        self.left_panel.addWidget(self.map_plot)

    def _init_polar_plot(self):
        self.polar_plot.setAspectLocked(True)
        self.polar_plot.hideAxis('left')
        self.polar_plot.hideAxis('bottom')
        for angle in np.arange(0, 360, 30):
            x = [0, 1.1 * np.cos(np.radians(angle))]
            y = [0, 1.1 * np.sin(np.radians(angle))]
            self.polar_plot.addItem(pg.PlotCurveItem(x, y, pen=pg.mkPen('#404040', width=0.5)))
        for r in np.linspace(0.25, 1.0, 4):
            theta = np.linspace(0, 2 * np.pi, 100)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            self.polar_plot.addItem(pg.PlotCurveItem(x, y, pen=pg.mkPen('#404040', width=0.5)))

    def toggle_view(self):
        self.current_view = "polar" if self.current_view == "cartesian" else "cartesian"
        self.plot_stack.setCurrentIndex(0 if self.current_view == "cartesian" else 1)

    def create_antenna_array(self):
        angles = np.linspace(0, 2 * np.pi, self.num_antennas, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)]) * (self.wavelength / 2)

    def generate_signals(self):
        # Генерация случайных координат абонентов
        self.source_distances = np.random.uniform(100, self.range_limit, self.num_sources)
        self.source_angles = np.random.uniform(0, 360, self.num_sources)

        steering_vectors = np.exp(-1j * 2 * np.pi * (
                self.antenna_pos[:, 0, None] * np.cos(np.radians(self.source_angles)) +
                self.antenna_pos[:, 1, None] * np.sin(np.radians(self.source_angles))
        ) / self.wavelength)

        t = np.linspace(0, 1e-6, 1024)
        sources = np.random.randn(self.num_sources, len(t)) + 1j * np.random.randn(self.num_sources, len(t))
        signals = steering_vectors @ sources

        noise = 10 ** (-self.snr_db / 20) * (np.random.randn(*signals.shape) + 1j * np.random.randn(*signals.shape))
        return signals + noise

    def music_algorithm(self, signals):
        R = signals @ signals.conj().T / signals.shape[1]
        eigvals, eigvecs = np.linalg.eigh(R)
        noise_subspace = eigvecs[:, :-self.num_sources]

        theta_range = np.linspace(0, 360, 360)
        spectrum = np.zeros_like(theta_range)

        for i, theta in enumerate(theta_range):
            a = np.exp(-1j * 2 * np.pi * (
                    self.antenna_pos[:, 0] * np.cos(np.radians(theta)) +
                    self.antenna_pos[:, 1] * np.sin(np.radians(theta))
            ) / self.wavelength)
            spectrum[i] = 1 / np.linalg.norm(noise_subspace.conj().T @ a) ** 2

        peaks, _ = find_peaks(spectrum, height=0.5 * np.max(spectrum), distance=10)
        estimated_angles = theta_range[peaks][:self.num_sources]  # Ограничение по числу источников

        return estimated_angles, theta_range, spectrum / self.max_spectrum

    def update_map(self, estimated_angles):
        # """Обновление карты местности"""
        # for i in range(self.num_sources):
        #     # Преобразование полярных координат в декартовы
        #     x = self.source_distances[i] * np.cos(np.radians(self.source_angles[i]))
        #     y = self.source_distances[i] * np.sin(np.radians(self.source_angles[i]))
        #
        #     # Отрисовка линии направления
        #     self.target_lines[i].setData(
        #         [0, x],
        #         [0, y],
        #         pen=pg.mkPen(['#FF0000', '#00FF00', '#0000FF'][i], width=2)
        # #     )
        # """Обновление карты: линии, маркеры и подписи"""
        # positions = []
        # display_distance = self.range_limit * 0.8
        #
        # # Расчет координат и обновление элементов
        # for i in range(self.num_sources):
        #     # Полярные -> декартовы
        #     angle_rad = np.radians(self.source_angles[i])
        #     x = self.source_distances[i] * np.cos(angle_rad)
        #     y = self.source_distances[i] * np.sin(angle_rad)
        #     positions.append((x, y))
        #
        #     # Линии направления
        #     if i < len(self.target_lines):
        #         self.target_lines[i].setData([0, x], [0, y],
        #                                      pen=pg.mkPen(['#FF0000', '#00FF00', '#0000FF'][i], width=2))
        #
        #     # Маркеры и подписи
        #     if i < len(self.target_markers):
        #         self.target_markers[i].setData([x], [y])
        #         self.target_labels[i].setPos(x, y)
        #
        # # Сброс неиспользуемых элементов
        # for j in range(len(positions), len(self.target_markers)):
        #     self.target_markers[j].setData([], [])
        #     self.target_labels[j].setPos(-1000, -1000)
        # display_distance = self.range_limit * 0.8
        # valid_angles = estimated_angles[:self.num_sources]  # Обрезаем лишние
        #
        # # Обновление линий направлений
        # for i, line in enumerate(self.target_lines):
        #     if i < len(valid_angles):
        #         angle = valid_angles[i]
        #         x = display_distance * np.cos(np.radians(angle))
        #         y = display_distance * np.sin(np.radians(angle))
        #         line.setData([0, x], [0, y], pen=self.target_lines[i].pen)
        #     else:
        #         line.setData([], [])
        #
        # # Реальные позиции (остаются неизменными)
        # for i in range(self.num_sources):
        #     x_real = self.source_distances[i] * np.cos(np.radians(self.source_angles[i]))
        #     y_real = self.source_distances[i] * np.sin(np.radians(self.source_angles[i]))
        #     self.target_markers[i].setData([x_real], [y_real])
        """Обновление карты: линии - вычисленные направления, маркеры - реальные позиции"""
        display_distance = self.range_limit * 0.8  # 80% радиуса карты

        # Очистка предыдущих направлений
        for line in self.target_lines:
            line.setData([], [])

        # Рисование новых направлений
        for i, angle in enumerate(estimated_angles):
            x = display_distance * np.cos(np.radians(angle))
            y = display_distance * np.sin(np.radians(angle))
            color = ['#FF0000', '#00FF00', '#0000FF'][i % 3]

            self.target_lines[i].setData(
                [0, x],
                [0, y],
                pen=pg.mkPen(color, width=2, style=QtCore.Qt.PenStyle.DashLine)
            )

        # Обновление реальных позиций абонентов
        for i in range(self.num_sources):
            x_real = self.source_distances[i] * np.cos(np.radians(self.source_angles[i]))
            y_real = self.source_distances[i] * np.sin(np.radians(self.source_angles[i]))
            self.target_markers[i].setData([x_real], [y_real])
            self.target_labels[i].setPos(x_real, y_real)

    def update(self):
        signals = self.generate_signals()
        estimated_angles, theta_range, spectrum = self.music_algorithm(signals)

        # Обновление графиков
        self.cartesian_curve.setData(theta_range, spectrum)
        for i, m in enumerate(self.cartesian_markers):
            m.setPos(self.source_angles[i] if i < len(self.source_angles) else 0)

        theta_rad = np.radians(theta_range)
        x = spectrum * np.cos(theta_rad)
        y = spectrum * np.sin(theta_rad)
        self.polar_curve.setData(x, y)

        # Обновление карты
        self.update_map(estimated_angles)

    def run(self):
        self.win.show()
        self.app.exec()


if __name__ == "__main__":
    system = AdvancedDOASystem()
    system.run()