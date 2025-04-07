# subscriber_manager.py
from dmr_subscriber import DMRSubscriber
from strategy import DirectModeSimulationStrategy
from PyQt5.QtCore import Qt, QTimer
import logging

class SubscriberManager:
    """Управляет списком абонентов."""
    def __init__(self, tdma):
        self.subscribers = []
        self.tdma = tdma  # Сохраняем ссылку на TDMA
        transmit_timer = QTimer()
        transmit_timer.timeout.connect(self.transmit_messages)
        transmit_timer.start(50)  # Каждые полсекунды
        self.logger = logging.getLogger(__name__)

    def add_subscriber(self, latitude, longitude, carrier_frequency):
        """Добавляет нового абонента."""
        new_id = len(self.subscribers) + 1
        new_subscriber = DMRSubscriber(id=new_id,
                                         latitude=latitude,
                                         longitude=longitude,
                                         simulation_strategy=DirectModeSimulationStrategy(),
                                         carrier_frequency=carrier_frequency,
                                         tdma=self.tdma)
        slot = self.tdma.assign_slot(new_id)  # Назначаем слот
        self.logger.debug(f"Assigned slot {slot} to subscriber {new_id}")  # Добавляем логгирование
        self.subscribers.append(new_subscriber)
        return new_subscriber

    def remove_subscriber(self, subscriber):
        """Удаляет абонента."""
        self.subscribers.remove(subscriber)

    def get_subscribers(self):
        """Возвращает список абонентов."""
        return self.subscribers

    def transmit_messages(self):
        print("transmit_messages")
        for sub in self.subscribers:
            print(f'sub {sub.id}')
            sub.generate_random_message(200) # Генерируем случайное сообщение
            sub.transmit()  #  Помещаем сигнал в буфер
