import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import numpy as np
import subscribers as sub

class Visualizer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.direction_lines = []
        self.init_ui()

    def init_ui(self):
        self.layout = QtWidgets.QHBoxLayout(self)

        # Левая панель: графики
        self.left_panel = QtWidgets.QVBoxLayout()
        self.init_cartesian_plot()
        self._init_spectrum_plot()
        self.layout.addLayout(self.left_panel)

        # Правая панель: карта
        self.right_panel = QtWidgets.QVBoxLayout()
        self._init_map_plot()
        self.layout.addLayout(self.right_panel)

        # Таймер
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(500)

    def init_cartesian_plot(self):
        self.cartesian_plot = pg.PlotWidget(title="Пеленгация")
        self.cartesian_curve = self.cartesian_plot.plot(pen='y')
        self.cartesian_markers = [
            pg.InfiniteLine(pos=0, angle=90, pen=color)
            for color in ['#FF0000', '#00FF00', '#0000FF']
        ]
        for m in self.cartesian_markers:
            self.cartesian_plot.addItem(m)
        self.left_panel.addWidget(self.cartesian_plot)

    def _init_spectrum_plot(self):
        self.spectrum_plot = pg.PlotWidget(title="Спектральное полотно")
        self.spectrum_plot.setLabel("left", "Уровень, дБ")
        self.spectrum_plot.setLabel("bottom", "Частота, МГц")
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        self.spectrum_curve = self.spectrum_plot.plot(pen="#ffffff")
        self.spectrum_markers = []
        self.left_panel.addWidget(self.spectrum_plot)

    def _init_map_plot(self):
        self.map_plot = pg.PlotWidget(title="Карта местности")
        self.map_plot.setAspectLocked(True)
        self.map_plot.setXRange(-1200, 1200)
        self.map_plot.setYRange(-1200, 1200)
        self.map_plot.showGrid(x=True, y=True, alpha=0.3)

        # Центральный маркер (пеленгатор)
        self.doa_marker = pg.ScatterPlotItem(
            pos=[(0, 0)],
            symbol='s',
            size=20,
            pen=pg.mkPen('#FFA500'),
            brush=pg.mkBrush('#FFA500')
        )
        self.map_plot.addItem(self.doa_marker)

        # Маркеры целей
        self.target_markers = [
            pg.ScatterPlotItem(
                symbol='o',
                size=15,
                pen=pg.mkPen(color),
                brush=pg.mkBrush(color),
                name=f"Цель {i + 1}"
            )
            for i, color in enumerate(['#FF0000', '#00FF00', '#0000FF'])
        ]
        for marker in self.target_markers:
            self.map_plot.addItem(marker)

        # Линии направлений (изначально скрыты)
        self.direction_lines = [
            pg.PlotCurveItem(
                pen=pg.mkPen(color, width=2, style=QtCore.Qt.PenStyle.DashLine)
            )
            for color in ['#FF0000', '#00FF00', '#0000FF']
        ]
        for line in self.direction_lines:
            line.setZValue(1)
            self.map_plot.addItem(line)

        # Текстовые метки
        self.target_labels = [
            pg.TextItem(
                text=f"Цель {i + 1}",
                color=color,
                anchor=(0.5, 1.2),
                # font=QtGui.QFont('Arial', 12)
            )
            for i, color in enumerate(['#FF0000', '#00FF00', '#0000FF'])
        ]

        for label in self.target_labels:
            self.map_plot.addItem(label)

        self.right_panel.addWidget(self.map_plot)

    def update_plots(self, angles, spectrum, freqs, spectrum_db, subscribers: list):
        # Декартов график
        self.cartesian_curve.setData(np.linspace(0, 360, len(spectrum)), spectrum)
        for i, m in enumerate(self.cartesian_markers):
            m.setPos(angles[i] if i < len(angles) else 0)

        # Спектр
        # self.spectrum_curve.setData(freqs, spectrum_db)
        self.update_map(subscribers)
        self.update_spectrum(freqs, spectrum_db, subscribers)

    def update_map(self, subscribers):
        center = (0, 0)  # Позиция пеленгатора
        for i, sub in enumerate(subscribers):
            # Расчет координат
            x, y = sub.coordinates

            # Обновление линии направления
            if i < len(self.direction_lines):
                self.direction_lines[i].setData(
                    [center[0], x],
                    [center[1], y],
                    pen=pg.mkPen(sub.color, width=1.5, style=QtCore.Qt.PenStyle.DashLine)
                )
                self.direction_lines[i].show()

            # Обновление маркера
            self.target_markers[i].setData([x], [y])

            # Обновление метки
            self.target_labels[i].setPos(x, y)
            self.target_labels[i].setText(
                f"Цель {i + 1}\n"
                f"Дистанция: {sub.distance:.0f} м\n"
                f"Азимут: {sub.angle:.1f}°\n"
                f"{sub.signal_type}"
            )

        # Скрытие неиспользуемых элементов
        for j in range(len(subscribers), len(self.target_markers)):
            self.target_markers[j].setData([], [])
            self.target_labels[j].setText("")

        # Скрытие неиспользуемых линий
        for j in range(len(subscribers), len(self.direction_lines)):
            self.direction_lines[j].hide()

    def update_spectrum(self, freqs_mhz: np.ndarray, spectrum_db: np.ndarray, subscribers: list[sub.Subscriber]):
        """Обновление спектра с маркерами абонентов"""
        # Обновление кривой спектра
        self.spectrum_curve.setData(freqs_mhz, spectrum_db)

        # Удаление старых маркеров
        for item in self.spectrum_markers:
            self.spectrum_plot.removeItem(item)
        self.spectrum_markers.clear()

        # Добавление новых маркеров
        for sub in subscribers:
            self._add_spectrum_marker(sub, spectrum_db)

    def _add_spectrum_marker(self, subscriber: sub.Subscriber, spectrum_db: np.ndarray):
        """Добавление маркера для одного абонента"""
        freq_mhz = subscriber.frequency / 1e6
        color = subscriber.color

        # Вертикальная линия
        line = pg.InfiniteLine(
            pos=freq_mhz,
            angle=90,
            pen=pg.mkPen(color, width=1, style=QtCore.Qt.PenStyle.DashLine)
        )

        # Текстовая метка
        text = pg.TextItem(
            text=f"{subscriber.signal_type.value}\n{freq_mhz:.1f} МГц",
            color=color,
            anchor=(0.5, 1),
            border=pg.mkPen(color),
            fill=pg.mkBrush("#00000080")
        )
        text.setPos(freq_mhz, np.max(spectrum_db) - 10)

        self.spectrum_plot.addItem(line)
        self.spectrum_plot.addItem(text)
        self.spectrum_markers.extend([line, text])


    def update(self):
        # Будет вызываться из главного приложения
        pass