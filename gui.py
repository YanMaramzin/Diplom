from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QListWidget, QListWidgetItem,
                             QSizePolicy, QLineEdit, QGraphicsRectItem, QGraphicsLineItem)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QBrush
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

    def set_subscribers(self, subscribers):
        """Устанавливает список абонентов и обновляет карту."""
        self.subscribers = subscribers
        self.clear()
        for subscriber in subscribers:
            self.add_subscriber(subscriber)

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

    def subscriber_clicked(self, item, points):
        """Обработчик клика на абоненте."""
        subscriber = item.subscriber  # Получаем абонента, связанного с маркером
        self.main_window.subscriber_info.set_subscriber(subscriber)  # Обновляем информацию

class SpectrumView(pg.PlotWidget):
    """Виджет для отображения спектра."""
    def __init__(self, sample_rate, main_window):
        super().__init__()
        self.setBackground('w')
        self.pen = pg.mkPen(color='b', width=2)
        self.x = []
        self.y = []
        self.curve = self.plot(self.x, self.y, pen=self.pen)
        self.setYRange(0, 1)
        self.showGrid(x=True, y=True, alpha=0.5)
        self.sample_rate = sample_rate # Добавляем sample_rate
        self.freq_lines = []  # Список линий
        self.main_window = main_window

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
        self.curve.setData(self.x / 1e6, self.y)

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
        self.mapView = MapView(self)
        self.spectrumView = SpectrumView(self.sample_rate, self) # Передаем sample_rate

        # TDMA parameters
        self.num_slots = 4
        self.frame_duration = 0.1  # seconds
        self.tdma = TDMA(self.num_slots, self.frame_duration)

        # Контроллеры
        self.subscriber_manager = SubscriberManager(self.tdma) #  Передаем TDMA
        self.map_controller = MapController(self.mapView)
        self.spectrum_controller = SpectrumController(self.spectrumView, self.sample_rate)

        self.init_ui()
        self.mapView.set_subscribers(self.subscribers)
        self.populate_subscriber_list()  # Заполняем список сразу после создания
        self.logger = logging.getLogger(__name__)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(100)  # Обновление каждые 100 мс (можно изменить)

    def init_ui(self):
        # Основной layout (теперь вертикальный)
        mainLayout = QVBoxLayout()

        # Верхняя панель (информация и управление)
        topPanel = QHBoxLayout()

        # 1. Число абонентов
        self.numSubscribersLabel = QLabel(f"Number of Subscribers: {len(self.subscribers)}")
        topPanel.addWidget(self.numSubscribersLabel)

        # 2. Кнопка "Add Subscriber"
        self.addSubscriberButton = QPushButton("Add Subscriber")
        self.addSubscriberButton.clicked.connect(self.add_subscriber)
        topPanel.addWidget(self.addSubscriberButton)

        # Список абонентов
        self.subscriberListWidget = QListWidget()
        self.subscriberListWidget.itemClicked.connect(self.show_subscriber_details)
        self.update_subscriber_list() # Заполняем список абонентов
        leftPanelWidget = QWidget() #Чтобы растянуть список по вертикали
        listLayout = QVBoxLayout()
        listLayout.addWidget(self.subscriberListWidget)
        leftPanelWidget.setLayout(listLayout)
        leftPanelWidget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)  # Растягиваем по вертикали
        self.subscriberListWidget.itemClicked.connect(self.subscriber_list_item_clicked)  # Подключаем сигнал
        topPanel.addWidget(leftPanelWidget)

        # Панель графиков (Карта и Спектр)
        graphPanel = QHBoxLayout()
        # Карта
        # self.update_map() # Отображаем абонентов на карте
        graphPanel.addWidget(self.mapView)

        # Subscriber Info Widget
        self.subscriber_info = SubscriberInfoWidget()
        topPanel.addWidget(self.subscriber_info)

        # Спектр
        graphPanel.addWidget(self.spectrumView)

        self.tdma_view = TDMAView(self.num_slots, self.frame_duration)
        graphPanel.addWidget(self.tdma_view)

        # Добавляем панели в основной layout
        mainLayout.addLayout(topPanel)
        mainLayout.addLayout(graphPanel)
        self.setLayout(mainLayout)

        # Элементы управления для задания диапазона частот (МГц)
        frequency_range_layout = QHBoxLayout()

        self.min_frequency_label = QLabel("Min Frequency (MHz):")
        frequency_range_layout.addWidget(self.min_frequency_label)

        self.min_frequency_input = QLineEdit()
        self.min_frequency_input.setText("0.0")  # Значение по умолчанию
        frequency_range_layout.addWidget(self.min_frequency_input)

        self.max_frequency_label = QLabel("Max Frequency (MHz):")
        frequency_range_layout.addWidget(self.max_frequency_label)

        self.max_frequency_input = QLineEdit()
        self.max_frequency_input.setText(
            str(self.sample_rate / 2e6))  # Значение по умолчанию (половина частоты дискретизации)
        frequency_range_layout.addWidget(self.max_frequency_input)

        self.set_frequency_range_button = QPushButton("Set Frequency Range")
        self.set_frequency_range_button.clicked.connect(self.set_frequency_range)
        frequency_range_layout.addWidget(self.set_frequency_range_button)

        # Добавляем layout с элементами управления в основной layout
        mainLayout.addLayout(frequency_range_layout)

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

    def populate_subscriber_list(self):
        """Заполняет QListWidget списком абонентов."""
        self.subscriberListWidget.clear() #  Очищаем список перед заполнением
        for subscriber in self.subscriber_manager.get_subscribers():
            item = QListWidgetItem(f"ID: {subscriber.id}, Lat: {subscriber.latitude:.2f}, Lon: {subscriber.longitude:.2f}")
            item.subscriber = subscriber  # Сохраняем ссылку на абонента
            self.subscriberListWidget.addItem(item)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()