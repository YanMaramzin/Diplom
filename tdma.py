import numpy as np
import logging

class TDMA:
    """Реализация TDMA."""
    def __init__(self, num_slots, frame_duration):
        self.num_slots = num_slots
        self.frame_duration = frame_duration
        self.slot_duration = frame_duration / num_slots
        self.slot_assignments = {} # Словарь: subscriber_id -> slot_number
        self.logger = logging.getLogger(__name__)

    def assign_slot(self, subscriber_id):
        """Назначает слот абоненту."""
        #  Простой алгоритм назначения: первый свободный слот
        for slot in range(self.num_slots):
            if slot not in self.slot_assignments.values():
                self.slot_assignments[subscriber_id] = slot
                return slot
        return None # Нет свободных слотов

    def get_slot_for_subscriber(self, subscriber_id):
        """Возвращает слот, назначенный абоненту."""
        return self.slot_assignments.get(subscriber_id)

    def is_active_slot(self, subscriber_id, current_time):
        """Проверяет, активен ли слот для данного абонента в данный момент времени."""
        self.logger.debug(f"is_active_slot")
        slot = self.get_slot_for_subscriber(subscriber_id)
        if slot is None:
            self.logger.debug(f"Subscriber {subscriber_id} - Slot not assigned")
            return False  # Слот не назначен

        #  Берем остаток от деления current_time на frame_duration
        self.logger.debug(f"Current time before: {current_time}")
        current_time = current_time % self.frame_duration
        self.logger.debug(f"Current time after: {current_time}")

        slot_start_time = slot * self.slot_duration
        slot_end_time = slot_start_time + self.slot_duration
        result = slot_start_time <= current_time < slot_end_time
        self.logger.debug(
            f"Subscriber {subscriber_id} - Slot: {slot}, Start: {slot_start_time:.2f}, End: {slot_end_time:.2f}, Current: {current_time:.2f}, Active: {result}")
        return result

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