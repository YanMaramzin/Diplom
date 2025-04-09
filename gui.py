from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QListWidget, QListWidgetItem,
                             QGraphicsRectItem, QSplitter)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QBrush, QPen
import pyqtgraph as pg
import numpy as np
from signal_processing import calculate_spectrum, generate_dmr_signal, generate_noise_spectrum
from subscriber_manager import SubscriberManager #  Импортируем SubscriberManager
from map_controller import MapController #  Импортируем MapController
from spectrum_controller import SpectrumController #  Импортируем SpectrumController
import random
from tdma import TDMA
import logging

class TDMAView(QWidget):
    """Виджет для отображения TDMA-слотов."""
    def __init__(self, num_slots, frame_duration):
        super().__init__()
        self.num_slots = num_slots
        self.frame_duration = frame_duration
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.plot = pg.PlotWidget()
        self.plot.setXRange(0, frame_duration)
        self.plot.setYRange(0, num_slots)
        self.plot.setLabel('bottom', 'Time', units='s')
        self.plot.setLabel('left', 'Slot')
        self.plot.showGrid(x=True, y=True, alpha=0.5)
        self.layout.addWidget(self.plot)
        self.rects = {}
        self.dt = 0.01
        self.logger = logging.getLogger(f"TDMAView")  # Создаем логгер для каждого абонента
        self.transmission_states = {}  # Создаем словарь для хранения состояния передачи

    def update_slots(self, subscribers):
        self.logger.debug(f"update_slots")
        """Обновляет график TDMA-слотов, отображая активные слоты для абонентов."""
        # Очищаем предыдущие прямоугольники
        for rect in self.rects:
            self.plot.removeItem(rect)
        self.rects = {}

        # Вычисляем состояние передачи для каждого абонента на каждом шаге времени
        for subscriber in subscribers:
            slot = subscriber.tdma.get_slot_for_subscriber(subscriber.id)
            if slot is not None:
                current_time = subscriber.current_time  # Получаем текущее время абонента
                x = current_time % self.frame_duration  # Используем остаток от деления

                # Если прямоугольник для этого абонента еще не создан, создаем его
                if subscriber.id not in self.rects:
                    self.rects[subscriber.id] = {}

                while x < self.frame_duration:
                    is_transmitting = subscriber.tdma.is_active_slot(subscriber.id, x)

                    rect_id = (subscriber.id, x)  # Уникальный идентификатор для прямоугольника

                    if is_transmitting:
                        # Если прямоугольник уже существует, обновляем его координаты
                        if rect_id in self.rects[subscriber.id]:
                            rect = self.rects[subscriber.id][rect_id]
                            rect.setRect((current_time + x) * 1000, slot, self.dt * 1000, 100)  # Обновляем координаты
                        else:
                            # Если прямоугольника нет, создаем его
                            rect = QGraphicsRectItem((current_time + x) * 1000, slot, self.dt * 1000, 100)  # x, y, width, height
                            rect.setBrush(QBrush(QColor(0, 0, 255)))  # Синий цвет
                            self.plot.addItem(rect)
                            self.rects[subscriber.id][rect_id] = rect
                    else:
                        # Если абонент не передает, и прямоугольник существует, удаляем его
                        if rect_id in self.rects[subscriber.id]:
                            rect = self.rects[subscriber.id].pop(rect_id)
                            self.plot.removeItem(rect)

                    x += self.dt

        self.logger.debug(f"Number of rectangles: {len(self.rects)}")  # Добавляем логгирование

class SubscriberInfoWidget(QWidget):
    """Виджет для отображения информации об абоненте."""
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.labels = {}
        self.setMinimumSize(200, 100)  # Установите минимальный размер

    def set_subscriber(self, subscriber):
        """Отображает информацию о выбранном абоненте."""
        print(f"set_subscriber called with subscriber: {subscriber}")
        # Очищаем предыдущую информацию
        for label in self.labels.values():
            label.setParent(None)  # Удаляем виджет из layout
        self.labels = {}

        if subscriber:
            # Создаем метки для отображения параметров
            self.add_info("ID", subscriber.id)
            self.add_info("Latitude", f"{subscriber.latitude:.2f}")
            self.add_info("Longitude", f"{subscriber.longitude:.2f}")
            self.add_info("Carrier Frequency (MHz)", f"{subscriber.CARRIER_FREQ / 1e6:.3f}")
            # Добавьте другие параметры, которые вы хотите отображать
        else:
            # Если абонент не выбран, отображаем сообщение
            no_selection_label = QLabel("No subscriber selected")
            self.layout.addWidget(no_selection_label)
            self.labels["no_selection"] = no_selection_label

    def add_info(self, name, value):
        """Добавляет информацию в виджет."""
        label = QLabel(f"<b>{name}:</b> {value}")
        self.layout.addWidget(label)
        self.labels[name] = label # Сохраняем ссылку на label

class MapView(pg.PlotWidget):
    """Виджет для отображения карты с абонентами."""
    def __init__(self, main_window):
        super().__init__()
        self.setBackground('w')
        self.setXRange(-180, 180)
        self.setYRange(-90, 90)
        self.items = []
        self.texts = []
        self.subscribers = []
        self.showGrid(x=True, y=True, alpha=0.5)
        self.main_window = main_window  # Сохраняем ссылку на главное окно
        self.direction_finder_position = (0, 0)  # Пример координат
        self.direction_finder_item = None
        self.direction_lines = {}  # Словарь для хранения линий направления

        self.add_direction_finder()  # Добавляем пеленгатор при инициализации

    def add_direction_finder(self):
        """Добавляет пеленгатор на карту."""
        #  Задаем координаты пеленгатора,
        #  создаем элемент ScatterPlotItem для пеленгатора
        self.direction_finder_item = pg.ScatterPlotItem(
            pos=[self.direction_finder_position],
            size=15,
            brush=QBrush(QColor(0, 255, 0)),  # Зеленый цвет
            symbol='o'  # Круг
        )
        self.addItem(self.direction_finder_item)

    def set_subscribers(self, subscribers):
        """Устанавливает список абонентов и обновляет карту."""
        self.subscribers = subscribers
        self.clear()
        self.add_direction_finder()  # Перерисовываем пеленгатор
        for subscriber in subscribers:
            self.add_subscriber(subscriber)
            self.draw_direction_line(subscriber)  # Рисуем линию направления

    def add_subscriber(self, subscriber):
        """Добавляет абонента на карту."""
        longitude = subscriber.longitude
        latitude = subscriber.latitude

        item = pg.ScatterPlotItem(pos=[(longitude, latitude)],
                                  size=10,
                                  brush=QBrush(QColor(0, 0, 255)))
        item.subscriber = subscriber  # Сохраняем ссылку на абонента
        item.sigClicked.connect(self.subscriber_clicked) # Подключаем сигнал клика
        self.addItem(item)
        self.items.append(item)

        # Добавляем текстовую метку с координатами
        text = pg.TextItem(text=f"({latitude:.2f}, {longitude:.2f})", color=(0, 0, 0))  # Черный цвет
        text.setPos(longitude + 2, latitude)  # Смещаем текст немного вправо
        self.addItem(text)
        self.texts.append(text) # Добавляем в список

    def clear(self):
        """Очищает карту от всех абонентов."""
        for item in self.items:
            self.removeItem(item)
        self.items = []

        #  Удаляем текстовые элементы
        for text in self.texts:
            self.removeItem(text)
        self.texts = []

        # Удаляем линии направления
        for line in self.direction_lines.values():
            self.removeItem(line)
        self.direction_lines = {}

    def subscriber_clicked(self, item, points):
        """Обработчик клика на абоненте."""
        subscriber = item.subscriber  # Получаем абонента, связанного с маркером
        self.main_window.subscriber_info.set_subscriber(subscriber)  # Обновляем информацию

    def calculate_bearing(self, subscriber):
        """Рассчитывает пеленг на абонента."""
        #  Получаем координаты абонента
        lon = subscriber.longitude
        lat = subscriber.latitude

        #  Получаем координаты пеленгатора
        df_lon, df_lat = self.direction_finder_position

        #  Преобразуем координаты в радианы
        df_lat_rad = np.radians(df_lat)
        df_lon_rad = np.radians(df_lon)
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        #  Вычисляем разницу долгот
        delta_lon = lon_rad - df_lon_rad

        #  Вычисляем пеленг
        y = np.sin(delta_lon) * np.cos(lat_rad)
        x = np.cos(df_lat_rad) * np.sin(lat_rad) - np.sin(df_lat_rad) * np.cos(lat_rad) * np.cos(delta_lon)
        bearing_rad = np.arctan2(y, x)

        #  Преобразуем пеленг в градусы
        bearing_deg = np.degrees(bearing_rad)
        bearing_deg = (bearing_deg + 360) % 360  # Нормализуем пеленг в диапазон [0, 360)

        return bearing_deg

    def draw_direction_line(self, subscriber):
        """Рисует линию направления на абонента."""
        #  Рассчитываем пеленг
        bearing = self.calculate_bearing(subscriber)

        #  Получаем координаты абонента
        lon = subscriber.longitude
        lat = subscriber.latitude

        #  Получаем координаты пеленгатора
        df_lon, df_lat = self.direction_finder_position

        #  Вычисляем конечную точку линии
        length = 500  # Длина линии
        end_lon = df_lon + length * np.sin(np.radians(bearing))
        end_lat = df_lat + length * np.cos(np.radians(bearing))

        #  Создаем линию
        line = pg.PlotDataItem(
            x=[df_lon, end_lon],
            y=[df_lat, end_lat],
            pen=QPen(QColor(0, 255, 0))  # Зеленый цвет, толщина 2
        )

        #  Добавляем линию на карту
        self.addItem(line)
        self.direction_lines[subscriber.id] = line  # Сохраняем линию в словаре

class SpectrumView(pg.PlotWidget):
    """Виджет для отображения спектра."""
    def __init__(self, sample_rate):
        super().__init__()
        self.setBackground('w')
        self.pen = pg.mkPen(color='b', width=2)
        self.x = []
        self.y = []
        self.curve = self.plot(self.x, self.y, pen=self.pen)
        self.showGrid(x=True, y=True, alpha=0.5)
        self.sample_rate = sample_rate # Добавляем sample_rate
        self.freq_lines = []  # Список линий

        self.min_frequency = 0
        self.max_frequency = sample_rate / 2  # По умолчанию - половина частоты дискретизации
        self.set_frequency_range(0, self.max_frequency)  # Установите диапазон частот при инициализации

    def update_spectrum(self, spectrum_data, frequencies, subscriber_frequencies=[]):
        """Обновляет график спектра и добавляет отметки частот."""
        #  Находим индексы частот, попадающих в диапазон

        start_index = np.argmin(np.abs(frequencies - self.min_frequency))
        end_index = np.argmin(np.abs(frequencies - self.max_frequency))

        #  Обрезаем данные
        frequencies = frequencies[start_index:end_index]
        spectrum_data = spectrum_data[start_index:end_index]

        # Если нет данных (нет абонентов в диапазоне), генерируем шум
        if len(spectrum_data) == 0:
            frequencies = np.linspace(self.min_frequency, self.max_frequency, 500)  #  Создаем массив частот
            noise_power = 0.01  #  Уровень шума
            spectrum_data = generate_noise_spectrum(frequencies, noise_power)

        self.x = frequencies
        self.y = spectrum_data
        self.curve.setData(self.x / 1e6, np.log10(self.y + 0.1))

        # Удаляем старые отметки
        # self.clear_frequency_markers()

        # Добавляем новые отметки
        # for freq in subscriber_frequencies:
        #   self.add_frequency_marker(freq / 1e6)

        # print("update_spectrum")

    def add_frequency_marker(self, frequency):
      """Добавляет вертикальную линию, отмечающую частоту."""
      line = pg.InfiniteLine(pos=frequency, angle=90, movable=False, pen=pg.mkPen('r'))
      self.addItem(line)
      self.freq_lines.append(line)

    def clear_frequency_markers(self):
      """Удаляет все отметки частот."""
      for line in self.freq_lines:
          self.removeItem(line)
      self.freq_lines = []

    def set_frequency_range(self, min_frequency, max_frequency):
        """Устанавливает диапазон отображаемых частот."""
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.setXRange(self.min_frequency / 1e6, self.max_frequency / 1e6)

class MainWindow(QWidget):
    """Главное окно приложения."""
    def __init__(self, subscribers, sample_rate):
        super().__init__()
        self.setWindowTitle("DMR Subscriber Map and Spectrum")
        self.setGeometry(100, 100, 1000, 700)
        self.subscribers = subscribers  # Передаем список абонентов
        self.sample_rate = sample_rate

        # TDMA parameters
        self.num_slots = 4
        self.frame_duration = 0.1  # seconds
        self.tdma = TDMA(self.num_slots, self.frame_duration)

        self.subscriber_manager = SubscriberManager(self.tdma)  # Передаем TDMA

        self.init_ui()
        # Контроллеры
        self.map_controller = MapController(self.map)
        self.spectrum_controller = SpectrumController(self.spectrum_view, self.sample_rate)

        self.map.set_subscribers(self.subscribers)
        # self.populate_subscriber_list()  # Заполняем список сразу после создания
        self.logger = logging.getLogger(__name__)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(100)  # Обновление каждые 100 мс (можно изменить)

    def init_ui(self):
        self.map = MapView(self.subscriber_manager)
        # self.map.setFixedSize(400, 400)

        #  График спектра
        self.spectrum_view = SpectrumView(self.sample_rate)
        # self.spectrum_view.plot_widget.setLabel('bottom', 'Frequency', units='Hz')
        # self.spectrum_view.plot_widget.setLabel('left', 'Amplitude')

        #  Список абонентов и кнопка "Добавить"
        self.subscriber_list = QListWidget()
        self.add_subscriber_button = QPushButton("Добавить абонента")
        self.add_subscriber_button.clicked.connect(self.add_subscriber)
        subscriber_layout = QVBoxLayout()
        subscriber_layout.addWidget(self.add_subscriber_button)
        subscriber_layout.addWidget(self.subscriber_list)
        subscriber_widget = QWidget()
        subscriber_widget.setLayout(subscriber_layout)

        #  График TDMA-слотов
        self.tdma_view = TDMAView(self.num_slots, self.frame_duration)
        self.tdma_view.plot.setLabel('bottom', 'Time', units='s')
        self.tdma_view.plot.setLabel('left', 'Slot')
        tdma_layout = QVBoxLayout()
        tdma_layout.addWidget(self.tdma_view)
        tdma_widget = QWidget()
        tdma_widget.setLayout(tdma_layout)

        #  Создаем горизонтальный разделитель для карты и спектра
        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.addWidget(self.map)
        left_splitter.addWidget(self.spectrum_view)
        left_splitter.setSizes([300, 100])

        #  Создаем вертикальный разделитель для списка абонентов и TDMA
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(subscriber_widget)
        right_splitter.addWidget(tdma_widget)
        right_splitter.setSizes([100, 300])

        #  Создаем главный разделитель для левой и правой частей
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([300, 100])

        #  Создаем главный макет
        main_layout = QHBoxLayout()
        main_layout.addWidget(main_splitter)
        self.setLayout(main_layout)

        self.setWindowTitle("DMR Simulator")

        # # Основной layout (теперь вертикальный)
        # mainLayout = QVBoxLayout()
        #
        # # Верхняя панель (информация и управление)
        # topPanel = QHBoxLayout()
        #
        # # 1. Число абонентов
        # self.numSubscribersLabel = QLabel(f"Number of Subscribers: {len(self.subscribers)}")
        # topPanel.addWidget(self.numSubscribersLabel)
        #
        # # 2. Кнопка "Add Subscriber"
        # self.addSubscriberButton = QPushButton("Add Subscriber")
        # self.addSubscriberButton.clicked.connect(self.add_subscriber)
        # topPanel.addWidget(self.addSubscriberButton)
        #
        # # Список абонентов
        # self.subscriberListWidget = QListWidget()
        # self.subscriberListWidget.itemClicked.connect(self.show_subscriber_details)
        # self.update_subscriber_list() # Заполняем список абонентов
        # leftPanelWidget = QWidget() #Чтобы растянуть список по вертикали
        # listLayout = QVBoxLayout()
        # listLayout.addWidget(self.subscriberListWidget)
        # leftPanelWidget.setLayout(listLayout)
        # leftPanelWidget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)  # Растягиваем по вертикали
        # self.subscriberListWidget.itemClicked.connect(self.subscriber_list_item_clicked)  # Подключаем сигнал
        # topPanel.addWidget(leftPanelWidget)
        #
        # # Панель графиков (Карта и Спектр)
        # graphPanel = QHBoxLayout()
        # # Карта
        # # self.update_map() # Отображаем абонентов на карте
        # graphPanel.addWidget(self.mapView)
        #
        # # Subscriber Info Widget
        # self.subscriber_info = SubscriberInfoWidget()
        # topPanel.addWidget(self.subscriber_info)
        #
        # # Спектр
        # graphPanel.addWidget(self.spectrumView)
        #
        # self.tdma_view = TDMAView(self.num_slots, self.frame_duration)
        # graphPanel.addWidget(self.tdma_view)
        #
        # # Добавляем панели в основной layout
        # mainLayout.addLayout(topPanel)
        # mainLayout.addLayout(graphPanel)
        # self.setLayout(mainLayout)
        #
        # # Элементы управления для задания диапазона частот (МГц)
        # frequency_range_layout = QHBoxLayout()
        #
        # self.min_frequency_label = QLabel("Min Frequency (MHz):")
        # frequency_range_layout.addWidget(self.min_frequency_label)
        #
        # self.min_frequency_input = QLineEdit()
        # self.min_frequency_input.setText("0.0")  # Значение по умолчанию
        # frequency_range_layout.addWidget(self.min_frequency_input)
        #
        # self.max_frequency_label = QLabel("Max Frequency (MHz):")
        # frequency_range_layout.addWidget(self.max_frequency_label)
        #
        # self.max_frequency_input = QLineEdit()
        # self.max_frequency_input.setText(
        #     str(self.sample_rate / 2e6))  # Значение по умолчанию (половина частоты дискретизации)
        # frequency_range_layout.addWidget(self.max_frequency_input)
        #
        # self.set_frequency_range_button = QPushButton("Set Frequency Range")
        # self.set_frequency_range_button.clicked.connect(self.set_frequency_range)
        # frequency_range_layout.addWidget(self.set_frequency_range_button)
        #
        # # Добавляем layout с элементами управления в основной layout
        # mainLayout.addLayout(frequency_range_layout)

    def add_subscriber(self):
        """Добавляет нового абонента (сигнал должен прийти из основного приложения)."""
        #  emit a signal to the main app. For example
        #  self.subscriber_added.emit() #  нужно будет настроить этот сигнал
        """Добавляет нового абонента."""
        #  Получаем координаты из полей ввода
        try:
            latitude = random.uniform(-90, 90)
            longitude = random.uniform(-180, 180)
            carrier_frequency = 500e6
        except ValueError:
            print("Пожалуйста, введите корректные числа")
            return

        #  Добавляем абонента с помощью SubscriberManager
        new_subscriber = self.subscriber_manager.add_subscriber(latitude, longitude, carrier_frequency)
        # self.subscriber_manager.add_subscriber(latitude, longitude, 500e6)
        #  Обновляем карту и список
        self.map_controller.update_map(self.subscriber_manager.get_subscribers())
        self.populate_subscriber_list()

    def update_subscriber_list(self):
        """Обновляет список абонентов."""
        self.subscriberListWidget.clear()
        for sub in self.subscribers:
            item = QListWidgetItem(f"Subscriber {sub.id}")
            item.setData(Qt.UserRole, sub.id)
            self.subscriberListWidget.addItem(item)

    def show_subscriber_details(self, item):
        """Отображает детальную информацию об абоненте."""
        sub_id = item.data(Qt.UserRole)
        subscriber = next((sub for sub in self.subscribers if sub.id == sub_id), None)
        if subscriber:
           print(f"Details for Subscriber {subscriber.id}: Latitude={subscriber.latitude}, Longitude={subscriber.longitude}")

    def update_data(self):
        """Обновляет данные и карту."""
        for sub in self.subscriber_manager.get_subscribers():
            self.logger.debug(f"Before should_transmit: Subscriber {sub.id} - current_time = {sub.current_time}")
            sub.update(0.01)  #  Обновляем время абонента
            is_transmitting = sub.should_transmit()
            self.logger.debug(f"After should_transmit: Subscriber {sub.id} - is_transmitting = {is_transmitting}")

        self.spectrum_controller.update_spectrum(self.subscriber_manager.get_subscribers())
        self.map_controller.update_map(self.subscriber_manager.get_subscribers())
        self.tdma_view.update_slots(self.subscriber_manager.get_subscribers())  # Обновляем график TDMA

    def set_frequency_range(self):
        """Устанавливает диапазон отображаемых частот на графике спектра."""
        try:
            min_frequency_mhz = float(self.min_frequency_input.text())
            max_frequency_mhz = float(self.max_frequency_input.text())

            #  Преобразуем в Гц
            min_frequency = min_frequency_mhz * 1e6
            max_frequency = max_frequency_mhz * 1e6

            if min_frequency >= max_frequency:
                print("Минимальная частота должна быть меньше максимальной.")
                return

            self.spectrumView.set_frequency_range(min_frequency, max_frequency)
        except ValueError:
            print("Пожалуйста, введите корректные числа в формате X.X")

    def subscriber_list_item_clicked(self, item):
        """Обработчик клика по элементу списка абонентов."""
        subscriber = item.subscriber  # Получаем абонента
        self.subscriber_info.set_subscriber(subscriber)  # Обновляем информацию

    # def populate_subscriber_list(self):
    #     """Заполняет QListWidget списком абонентов."""
    #     self.subscriberListWidget.clear() #  Очищаем список перед заполнением
    #     for subscriber in self.subscriber_manager.get_subscribers():
    #         item = QListWidgetItem(f"ID: {subscriber.id}, Lat: {subscriber.latitude:.2f}, Lon: {subscriber.longitude:.2f}")
    #         item.subscriber = subscriber  # Сохраняем ссылку на абонента
    #         self.subscriberListWidget.addItem(item)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()