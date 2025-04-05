import numpy as np
import logging

class TDMAFrame:
    """Класс, представляющий TDMA кадр."""
    def __init__(self, num_slots, slot_duration):
        self.num_slots = num_slots
        self.slot_duration = slot_duration # В секундах
        self.slots = [None] * num_slots # None = слот свободен
        self.logger = logging.getLogger("TDMAFrame")

    def insert_signal(self, signal, slot_number):
        """Вставляет сигнал в указанный слот."""
        if 0 <= slot_number < self.num_slots:
             self.slots[slot_number] = signal
             self.logger.debug(f"Inserted signal into slot {slot_number}.")
        else:
            self.logger.error("Invalid slot number.")
            raise ValueError("Invalid slot number.")


    def get_slot_signal(self, slot_number):
        """Возвращает сигнал из указанного слота."""
        if 0 <= slot_number < self.num_slots:
            self.logger.debug(f"Retrieved signal from slot {slot_number}.")
            return self.slots[slot_number]
        else:
            self.logger.error("Invalid slot number.")
            raise ValueError("Invalid slot number.")

    def get_frame_signal(self):
      """Собирает сигналы из всех слотов в один кадр."""
      frame_signal = np.array([])
      for slot in self.slots:
         if slot is not None:
            frame_signal = np.concatenate((frame_signal, slot))
         else:
            # Если слот пустой, добавляем тишину (нулевые отсчеты)
            frame_signal = np.concatenate((frame_signal, np.zeros(int(self.slot_duration * DMRSubscriber.SAMPLE_RATE))))
      self.logger.debug("Assembled frame signal.")
      return frame_signal

class Repeater:
    """Класс, представляющий ретранслятор."""
    def __init__(self, num_slots, slot_duration):
        self.tdma_frame = TDMAFrame(num_slots, slot_duration)
        self.logger = logging.getLogger("Repeater")

    def receive(self, signal, slot_number):
        """Принимает сигнал от абонента и помещает его в TDMA кадр."""
        self.tdma_frame.insert_signal(signal, slot_number)
        self.logger.info(f"Received signal in slot {slot_number}.")

    def transmit(self):
        """Передает TDMA кадр."""
        frame_signal = self.tdma_frame.get_frame_signal()
        self.logger.info("Transmitting TDMA frame.")
        return frame_signal