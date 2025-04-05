import logging
from abc import ABC, abstractmethod
from enum import Enum
from channel_model import ChannelModel

class CommunicationModeEnum(Enum):
    DIRECT = "direct"
    REPEATER = "repeater"


class CommunicationMode(ABC):
    """Абстрактный базовый класс для режимов связи."""
    def __init__(self, mode: CommunicationModeEnum, channel: ChannelModel):
        self.mode = mode
        self.channel = channel  #  ChannelModel
        self.logger = logging.getLogger("CommunicationMode")

    @abstractmethod
    def transmit(self, subscriber, destination_subscriber=None):
        """Передает сигнал от абонента (абстрактный метод)."""
        raise NotImplementedError

    @abstractmethod
    def receive(self, signal, subscriber):
        """Принимает сигнал абонентом (абстрактный метод)."""
        raise NotImplementedError

class DirectMode(CommunicationMode):
    """Режим прямой связи."""
    def __init__(self, channel: ChannelModel):
        super().__init__(CommunicationModeEnum.DIRECT, channel)
        self.logger = logging.getLogger("DirectMode")

    def transmit(self, subscriber, destination_subscriber):
        """Передает сигнал напрямую другому абоненту."""
        signal = subscriber.transmit() # Получаем сигнал от абонента
        if signal is not None:
          transmitted_signal = self.channel.transmit(signal)  # Пропускаем через канал
          self.logger.info(f"Transmitting directly to subscriber {destination_subscriber.id} through {type(self.channel).__name__}")
          return transmitted_signal
        else:
          self.logger.warning("No signal to transmit.")
          return None


    def receive(self, signal, subscriber):
        """Принимает сигнал напрямую от другого абонента."""
        self.logger.info("Receiving direct signal.")
        return subscriber.receive(signal)


class RepeaterMode(CommunicationMode):
    """Режим связи через ретранслятор."""
    def __init__(self, repeater, slot_number, channel: ChannelModel):
        super().__init__(CommunicationModeEnum.REPEATER, channel)
        self.repeater = repeater
        self.slot_number = slot_number
        self.logger = logging.getLogger("RepeaterMode")

    def transmit(self, subscriber):
        """Передает сигнал ретранслятору."""
        signal = subscriber.transmit()
        if signal is not None:
            transmitted_signal = self.channel.transmit(signal) # Пропускаем через канал
            self.logger.info(f"Subscriber {subscriber.id} transmitting to repeater in slot {self.slot_number} through {type(self.channel).__name__}")
            return transmitted_signal, self.slot_number  # Возвращаем и сигнал, и номер слота
        else:
            self.logger.warning(f"Subscriber {subscriber.id} has no signal to transmit to repeater.")
            return None, None

    def receive(self, signal, subscriber):
        """Принимает сигнал от ретранслятора."""
        self.logger.info("Receiving from repeater.")
        return subscriber.receive(signal)