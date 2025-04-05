from abc import ABC, abstractmethod
from enum import Enum

class EventType(Enum):
    """Типы событий для Observer Pattern."""
    TRANSMIT = "transmit"
    RECEIVE = "receive"
    DEMODULATE = "demodulate"

class Observer(ABC):
    """Абстрактный класс Observer."""
    @abstractmethod
    def update(self, event_type: EventType, data):
        """Реагирует на события."""
        pass

class Observable(ABC):
    """Абстрактный класс Observable."""
    def __init__(self):
        self._observers = []

    def attach(self, observer: Observer):
        """Подписывает наблюдателя."""
        self._observers.append(observer)

    def detach(self, observer: Observer):
        """Отписывает наблюдателя."""
        self._observers.remove(observer)

    def notify(self, event_type: EventType, data):
        """Уведомляет всех наблюдателей о событии."""
        for observer in self._observers:
            observer.update(event_type, data)