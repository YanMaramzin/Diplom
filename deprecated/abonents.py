from abc import ABC, abstractmethod
import random
import logging
from time import sleep

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import queue
import threading
import time
from typing import Dict, Optional

from pyexpat.errors import messages

# Настройка логгера
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Конфигурация DMR
SYMBOL_RATE = 4800
SYMBOL_DURATION = 1 / SYMBOL_RATE
FS = 48000  # Частота дискретизации (рекомендуется 48 кГц для DMR)
DEVIATION = 648

DMR_FREQUENCIES = {
    '00': 1.5*DEVIATION,
    '01': 3.0*DEVIATION,
    '10': -1.5*DEVIATION,
    '11': -3.0*DEVIATION
}

class DMRSyncTYPE(Enum):
    DATA = 0,
    VOICE = 1

class DMRStructureError(Exception):
    pass

class RadioChannel(ABC):
    def __init__(self):
        self.abonents: Dict[int, 'Abonent'] = {}
        self._running = False
        self._frame_counter = 0
        self.slot_duration = 30e-3

    def add_abonent(self, abonent: 'Abonent'):
        if abonent.id in self.abonents:
            raise ValueError(f"Abonent ID {abonent.id} exists")

        logger.info("add_abonent")
        self.abonents[abonent.id] = abonent
        abonent.channel = self

    def start(self):
        self._running = True
        self._main_loop = threading.Thread(target=self._run)
        self._main_loop.start()

    def stop(self):
        self._running = False
        if self._main_loop.is_alive():
            self._main_loop.join()

    @abstractmethod
    def _run(self):
        pass

    def broadcast(self, signal: np.ndarray, sender: 'Abonent'):
        for abonent in self.abonents.values():
            if abonent.id != sender.id:
                abonent.handle_rx_signal(signal, sender)

class DMRChannel(RadioChannel):
    """Класс для эмуляции радиоканала между абонентами"""

    def __init__(self):
        super().__init__()
        self._current_slot = 1

    def _run(self):
        logger.info(f"run {time.time()}")
        while self._running:
            start_time = time.time()
            self._process_slot(self._current_slot)
            self._current_slot = 2 if self._current_slot == 1 else 1
            self._frame_counter += 1
            elapsed = time.time() - start_time
            time.sleep(max(0, self.slot_duration - elapsed))

    def _process_slot(self, slot_number: int):
        logger.info("_process_slot")
        for abonent in self.abonents.values():
            if isinstance(abonent, DMRAbonent):
                abonent.handle_tdma_slot(slot_number)

    # def add_abonent(self, abonent: 'DMRAbonent'):
    #     self.abonents[abonent.id] = abonent
    #     abonent.channel = self
    #
    # def start(self):
    #     self._running = True
    #     self._tdma_loop = threading.Thread(target=self._tdma_scheduler)
    #     self._tdma_loop.start()
    #
    # def stop(self):
    #     self._running = False
    #     self._tdma_loop.join()
    #
    # def _tdma_scheduler(self):
    #     """Цикл TDMA с чередованием слотов"""
    #     while self._running:
    #         start_time = time.time()
    #
    #         # Оповещаем всех абонентов о начале слота
    #         for abonent in self.abonents.values():
    #             abonent.handle_tdma_slot(self._current_slot)
    #
    #         # Переключаем слот
    #         self._current_slot = 2 if self._current_slot == 1 else 1
    #         self._frame_counter += 1
    #
    #         # Ждем до конца слота
    #         elapsed = time.time() - start_time
    #         time.sleep(max(0, self.slot_duration - elapsed))


def generate_random_bits(num_bits):
    """Генерирует случайную битовую строку четной длины"""
    if num_bits % 2 != 0:
        num_bits += 1  # Делаем длину четной
    return ''.join(random.choice('01') for _ in range(num_bits))

def generate_sync_pattern(sync_type: DMRSyncTYPE = DMRSyncTYPE.VOICE) -> str:
    """Генерация синхропоследовательности согласно стандарту DMR"""
    # Синхропоследовательности из спецификации ETSI TS 102 361-1
    sync_patterns = {
        DMRSyncTYPE.VOICE : '010101010101010101010101010101010101010101010101',
        DMRSyncTYPE.DATA : '110011000011110011000011110011000011110011000011'
    }
    return sync_patterns[sync_type]

def fsk_4_modulation(input_bits, frequencies, Ts, Fs):
    symbols = [input_bits[i:i+2] for i in range(0, len(input_bits), 2)]
    n_symbols = len(symbols)
    samples_per_symbol = int(Ts * Fs)
    total_samples = n_symbols * samples_per_symbol
    t = np.arange(total_samples) / Fs  # Временная ось

    signal = np.zeros(total_samples)
    for i, sym in enumerate(symbols):
        freq = frequencies.get(sym, 0)
        start = i * samples_per_symbol
        end = (i + 1) * samples_per_symbol
        signal[start:end] = np.sin(2 * np.pi * freq * t[start:end])

    return t, signal


def add_rf_effects(signal, fc=400e6, fs=48e3, noise_power=0.1):
    """
    Добавляет эффекты радиочастотного тракта:
    - fc: несущая частота (например, 400 МГц)
    - fs: частота дискретизации
    - noise_power: уровень шума
    """
    t = np.arange(len(signal)) / fs

    # Перенос на несущую частоту
    carrier = np.exp(1j * 2 * np.pi * fc * t)
    rf_signal = signal * carrier.real

    # Добавление шума
    noise = np.random.normal(0, noise_power, len(rf_signal))
    rf_signal += noise

    # Полосовая фильтрация (имитация фильтра RF)
    from scipy.signal import butter, lfilter
    b, a = butter(4, [fc - 5e3, fc + 5e3], 'bandpass', fs=fs)
    rf_signal = lfilter(b, a, rf_signal)

    return rf_signal

class Abonent(ABC):
    def __init__(self, abonent_id: int):
        self.id = abonent_id
        self.rx_queue = queue.Queue()
        self.tx_queue = queue.Queue()
        self.channel: Optional[RadioChannel] = None
        self.current_slot = 1

    @abstractmethod
    def _modulate(self, bits: str) -> np.ndarray:
        pass

    def _demodulate(self, signal: np.ndarray) -> str:
        pass

    def send_message(self, message: str, dest_id: int):
        logger.info("send_message")
        packet = {
            'source': self.id,
            'dest': dest_id,
            'data': message,
            'timestamp': time.time()
        }
        self.tx_queue.put(packet)
        logger.info("Размер очереди:" + str(self.tx_queue.qsize()))

    def receive_messages(self) -> list:
        logger.info("receive_messages")
        messages = []
        while not self.rx_queue.empty():
            logger.info("while")
            messages.append(self.rx_queue.get())
        return messages

    def handle_rx_signal(self, signal: np.ndarray, sender: 'Abonent'):
        bits = self._demodulate(signal)
        self._process_bits(bits, sender)

    def _process_bits(self, bits: str, sender: 'Abonent'):
        if len(bits) < 8:
            logger.info(f"Длина пакета меньше 8")
            return

        try:
            dest_bits = bits[:8]
            dest_id = int(dest_bits, 2)  # <-- Корректное преобразование
            logger.debug(f"Dest bits: {dest_bits} -> Dest ID: {dest_id}")
            if dest_id == self.id:
                data = self._bits_to_str(bits[8:])
                self.rx_queue.put({
                    'source': sender.id,
                    'data': data,
                    'timestamp': time.time()
                })
                logger.info(f"self.rx_queue.empty(): {self.rx_queue.empty()}")
            else:
                logger.info(f"self.rx_queue.empty(): {self.rx_queue.empty()}")
        except ValueError:
            logger.error(f"Error parsing dest_id: {e}")

    @staticmethod
    def _bits_to_str(bits: str) -> str:
        chars = []
        for i in range(0, len(bits), 8):
            byte = bits[i:i+8]
            try:
                chars.append(chr(int(byte, 2)))
            except:
                logger.info("EXCEPT")
                continue
        return ''.join(chars)

    # @abstractmethod
    # def _generate_slot(self):
    #     pass
    #
    # @abstractmethod
    # def _generate_frame(self):
    #     pass
    #
    # def generate_signal(self):
    #     """Основной метод генерации сигнала"""
    #     self.bits = self._generate_frame()
    #     self.symbols = self._bits_to_symbols()
    #     self.signal = self._modulate()
    #     return self.signal
    #
    # def _bits_to_symbols(self) -> list:
    #     return [self.bits[i:i+2] for i in range(0, len(self.bits), 2)]


# class DMRAbonent(Abonent):
#
#     NUMS_BIT_DATA = 108
#     SLOT_BITS = 108 * 2
#
#     def __init__(self, sync_type: DMRSyncTYPE = DMRSyncTYPE.VOICE):
#         super().__init__()
#         self.sync_type = sync_type
#
#     def _generate_slot(self):
#         print("Generate slot")
#         """Генерация одного таймслота DMR"""
#         # Структура слота: синхронизация + данные
#         sync = generate_sync_pattern(self.sync_type)
#         data = generate_random_bits(self.SLOT_BITS - len(sync))
#         return sync + data
#
#     def _generate_frame(self) -> str:
#         """Генерация полного кадра (2 слота)"""
#         slot1 = self._generate_slot()
#         slot2 = self._generate_slot()
#         return slot1 + slot2
#
#     def _modulate(self) -> np.ndarray:
#         """Модуляция 4-FSK"""
#         samples_per_symbol = int(FS * SYMBOL_DURATION)
#         total_samples = len(self.symbols) * samples_per_symbol
#         t = np.arange(total_samples) / FS
#
#         signal = np.zeros(total_samples)
#         phase = 0  # Для сохранения фазы между символами
#
#         for i, sym in enumerate(self.symbols):
#             freq = DMR_FREQUENCIES.get(sym, 0)
#             start = i * samples_per_symbol
#             end = (i + 1) * samples_per_symbol
#             phase += 2 * np.pi * freq * t[start:end]
#             signal[start:end] = np.sin(phase)
#             phase %= 2 * np.pi  # Предотвращение переполнения фазы
#
#         return signal

class DMRAbonent(Abonent):
    def __init__(self, abonent_id: int):
        super().__init__(abonent_id)
        self.sync_type = 'voice'

    def _modulate(self, bits: str) -> np.ndarray:
        symbols = [bits[i:i + 2] for i in range(0, len(bits), 2)]
        samples_per_symbol = int(FS * SYMBOL_DURATION)  # Теперь 20 отсчетов
        t = np.arange(len(symbols) * samples_per_symbol) / FS

        signal = np.zeros(len(t))
        for i, sym in enumerate(symbols):
            freq = DMR_FREQUENCIES[sym]
            start = i * samples_per_symbol
            end = start + samples_per_symbol
            signal[start:end] = np.sin(2 * np.pi * freq * t[start:end])

        return signal

    def _demodulate(self, signal: np.ndarray) -> str:
        bits = []
        samples_per_symbol = int(FS * SYMBOL_DURATION)
        target_freqs = [972, 1944, -972, -1944]

        for i in range(0, len(signal), samples_per_symbol):
            symbol_signal = signal[i:i + samples_per_symbol]
            if len(symbol_signal) < 10:
                bits.append('00')
                continue

            # Вычисление спектра с повышенной точностью
            fft = np.fft.rfft(symbol_signal)
            freqs = np.fft.rfftfreq(len(symbol_signal), 1 / FS)

            # Поиск ближайшей целевой частоты
            best_match = None
            min_diff = float('inf')
            for f in target_freqs:
                idx = np.abs(freqs - abs(f)).argmin()
                curr_diff = abs(freqs[idx] - abs(f))
                if curr_diff < min_diff and np.abs(fft[idx]) > 0.1 * np.max(np.abs(fft)):
                    min_diff = curr_diff
                    best_match = f

            # Определение битов
            if best_match is None:
                bits.append('00')
            else:
                bits.append({
                                972: '00',
                                1944: '01',
                                -972: '10',
                                -1944: '11'
                            }[best_match])

        return ''.join(bits)

    def _map_freq_to_bits(self, freq: float) -> str:
        if 1700 < freq < 2200:
            return '01'
        elif 800 < freq < 1200:
            return '00'
        elif -2200 < freq < -1700:
            return '11'
        elif -1200 < freq < -800:
            return '10'
        return '00'

    def _freq_to_bits(self, freq: float) -> str:

        # Находим ближайшую целевую частоту
        closest_freq = min(DMR_FREQUENCIES.keys(), key=lambda x: abs(x - freq))

        # Проверка попадания в допустимый диапазон
        if abs(closest_freq - freq) > 500:
            logger.warning(f"Неверная частота: {freq:.1f} Гц (ожидалось ±972/1944 Гц)")
            return '00'  # Значение по умолчанию

        return DMR_FREQUENCIES[closest_freq]

    def handle_tdma_slot(self, slot_number: int):
        logger.info("handle_tdma_slot")
        self.current_slot = slot_number
        if self._should_transmit():
            self._transmit()

    def _should_transmit(self) -> bool:
        return (self.id % 2) == (self.current_slot % 2)

    def _transmit(self):
        if not self.tx_queue.empty() and self.channel:
            packet = self.tx_queue.get()
            try:
                # Форматирование dest_id как 8-битной двоичной строки
                dest_bits = format(packet['dest'], '08b')  # <-- Исправлено здесь
                data_bits = self._str_to_bits(packet['data'])
                full_bits = dest_bits + data_bits
                signal = self._modulate(full_bits)
                self.channel.broadcast(signal, self)
            except KeyError as e:
                logger.error(f"Missing key in packet: {e}")

    @staticmethod
    def _str_to_bits(s: str) -> str:
        return ''.join(format(ord(c), '08b') for c in s)


class Repeater(DMRAbonent):
    def __init__(self, repeater_id: int):
        super().__init__(repeater_id)
        self.re_transmit_delay = 0.050

    def handle_rx_signal(self, signal: np.ndarray, sender: Abonent):
        if sender.id == self.id:
            return

        bits = self._demodulate(signal)
        packet = self._parse_bits(bits)

        if packet and packet['dest'] != self.id:
            threading.Thread(target=self._delayed_retransmit, args=(packet,)).start()

    def _parse_bits(self, bits: str) -> Optional[dict]:
        try:
            return {
                'source': int(bits[:8], 2),
                'dest': int(bits[8:16], 2),
                'data': self._bits_to_str(bits[16:])
            }
        except:
            return None

    def _delayed_retransmit(self, packet: dict):
        time.sleep(self.re_transmit_delay)
        packet['hops'] = packet.get('hops', 0) + 1
        self.send_message(packet['data'], packet['dest'])


def plot_signals(signal: np.ndarray, title: str):
    plt.figure(figsize=(14, 6))
    plt.plot(np.arange(len(signal)) / FS, signal)
    plt.title(f"{title} (Time Domain)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.xlim(0, 0.02)
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.specgram(signal, Fs=FS, NFFT=1024, cmap='viridis')
    plt.title(f"{title} (Spectrogram)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(-3000, 3000)
    plt.colorbar(label="Power (dB)")
    plt.show()



# def plot_signals(signal: np.ndarray, fs: int, title: str):
#     """Визуализация сигнала и спектрограммы"""
#     # Временная область
#     plt.figure(figsize=(14, 8))
#     plt.plot(np.arange(len(signal)) / fs, signal)
#     plt.title(f"{title} (Time Domain)")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Amplitude")
#     plt.grid(True)
#     plt.xlim(0, 0.01)  # Показать первые 10 мс
#     plt.show()
#
#     # Спектрограмма
#     plt.figure(figsize=(14, 8))
#     plt.specgram(signal, Fs=fs, NFFT=1024, noverlap=512, cmap='viridis')
#     plt.title(f"{title} (Spectrogram)")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Frequency (Hz)")
#     plt.ylim(-3000, 3000)
#     plt.colorbar(label="Power (dB)")
#     plt.show()
# def test_modulation_demodulation():
#     abonent = DMRAbonent(1)
#
#     # Тест с разными комбинациями
#     test_cases = [
#         ('01', '01'),  # 1944 Hz
#         ('11', '11'),  # -1944 Hz
#         ('0011', '0011'),
#         ('', '')  # Пустой ввод
#     ]
#
#     for input_bits, expected in test_cases:
#         signal = abonent._modulate(input_bits)
#         demod_bits = abonent._demodulate(signal)
#         assert demod_bits == expected, f"Failed: {input_bits} → {demod_bits}"
#
#     print("All tests passed!")


def test_modulation_demodulation():
    abonent = DMRAbonent(1)
    test_bits = '01' * 5  # 10 бит = 5 символов

    # Модуляция
    signal = abonent._modulate(test_bits)

    # # Визуализация спектра
    plt.figure(figsize=(12, 6))
    plt.magnitude_spectrum(signal, Fs=FS, scale='dB')
    plt.xlim(0, 3000)
    plt.title("Спектр модулированного сигнала (биты '01')")
    plt.xlabel("Частота (Гц)")
    plt.ylabel("Мощность (дБ)")
    plt.grid(True)
    plt.show()

    # Демодуляция
    demod_bits = abonent._demodulate(signal)
    print(f"Ожидаемые биты: {test_bits}")
    print(f"Демодулированные биты: {demod_bits}")
    assert test_bits == demod_bits, "Демодуляция не удалась!"

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_modulation_demodulation()

# if __name__ == "__main__":
#     test_modulation_demodulation()
    # channel = DMRChannel()
    #
    # abonent1 = DMRAbonent(1)
    # # abonent2 = DMRAbonent(2)
    # # repeater = Repeater(100)
    #
    # # Создаем тестовый сигнал
    # test_bits = '01' * 10  # Должен соответствовать 1944 Гц
    # test_signal = abonent1._modulate(test_bits)
    #
    # # Демодулируем сигнал
    # demodulated_bits = abonent1._demodulate(test_signal)
    # print(f"Demodulated bits: {demodulated_bits}")  # Ожидаем: '0101010101...'

    # Визуализация спектра
    # symbol = test_signal[:int(FS * SYMBOL_DURATION)]
    # plt.magnitude_spectrum(symbol, Fs=FS, scale='dB')
    # plt.xlim(-3000, 3000)
    # plt.title("Спектр символа '01' (1944 Гц)")
    # plt.show()

    # channel.add_abonent(abonent1)
    # channel.add_abonent(abonent2)
    # channel.add_abonent(repeater)
    #
    # channel.start()
    #
    # # Тестовая передача сообщения
    # abonent1.send_message("Hello DMR World!", 2)
    #
    # # Даем время на обработку
    # time.sleep(0.5)
    #
    # # Получение сообщений
    # messages = abonent2.receive_messages()
    # for msg in messages:
    #     print(f"[Abonent {abonent2.id}] Received from {msg['source']}: {msg['data']} (Hops: {msg.get('hops', 0)})")
    #
    # # Визуализация последнего сигнала
    # if messages:
    #     test_bits = f"{2:08b}" + ''.join(format(ord(c), '08b') for c in "Test")
    #     test_signal = abonent1._modulate(test_bits)
    #     plot_signals(test_signal, "DMR Signal Example")
    #
    # channel.stop()

    # dmr = DMRAbonent()
    # Ts = 1 / SYMBOL_RATE
    # Fs = 44100
    #
    # num_bits = 108
    # data = generate_random_bits(num_bits)
    # sync = generate_sync_pattern()
    # data2 = generate_random_bits(num_bits)
    # slot = data + sync + data2
    # print(f"Сгенерированные биты: {slot}")
    #
    # t, signal = fsk_4_modulation(slot, DMR_FREQUENCIES, Ts, Fs)
    #
    # # Визуализация сигнала
    # plt.figure(figsize=(12, 6))
    # plt.plot(t, signal)
    # plt.xlabel('Время (с)')
    # plt.ylabel('Амплитуда')
    # plt.title('4-FSK модуляция')
    # plt.grid(True)
    # plt.xlim(0, Ts * 20)  # Показать первые 4 символа
    # plt.show()
    #
    # # Спектрограмма
    # plt.figure(figsize=(12, 6))
    # plt.specgram(signal, Fs=Fs, NFFT=1024, noverlap=512, cmap='viridis')
    # plt.xlabel('Время (с)')
    # plt.ylabel('Частота (Гц)')
    # plt.title('Спектрограмма 4-FSK сигнала')
    # plt.colorbar(label='Мощность (дБ)')
    # plt.ylim(0, 5000)  # Ограничить диапазон частот для наглядности
    # plt.show()

