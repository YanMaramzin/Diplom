from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from typing import List, Dict
import queue
import threading
import time

class SignalType(Enum):
    DMR = 1
    TETRA = 2
    GSM = 3

class Subscriber(ABC):
    def __init__(self, distance: float, angle: float, power=1.0):
        self.distance = distance
        self.angle = angle
        self.power = power
        self.signal_type: SignalType
        self.frequency: float
        self.color: str
        self.modulation: str
        self._ptt = False
        self.rx_buffer = queue.Queue()

    # @abstractmethod
    # def generate_signal(self, num_samples: int, fs: float) -> np.ndarray:
    #     """Генерация сигнала абонента"""
    #     pass

    @property
    def coordinates(self) -> tuple:
        """Декартовы координаты в метрах"""
        x = self.distance * np.cos(np.radians(self.angle))
        y = self.distance * np.sin(np.radians(self.angle))
        return x, y

    # @abstractmethod
    # def transmit(self):
    #     pass
    #
    @abstractmethod
    def receive(self, signal: np.ndarray):
        pass

class DMRChannel:
    """Класс для эмуляции радиоканала прямого режима"""
    def __init__(self, freq: float = 446.00625e6):
        self.freq = freq
        self.subscribers: Dict[str, DMRSubscriber] = {}
        self.attenuation = 0.0001
        self.noise_level = 0.01
        self.rx_queue = queue.Queue()
        print(f"Канал создан на частоте {self.freq / 1e6} МГц")

    def register_subscriber(self, radio: 'DMRSubscriber'):
        self.subscribers[radio.subs_id] = radio
        print(f"Абонент {radio.subs_id} зарегистрирован в канале")

    def broadcast(self, signal: np.ndarray, sender: 'DMRSubscriber'):
        print(f"\nПередача от {sender.subs_id} началась")
        start_time = time.time()
        for sub_id, subscriber in self.subscribers.items():
            if sub_id != sender.subs_id:
                distance = np.linalg.norm(
                    np.array(sender.coordinates) -
                    np.array(subscriber.coordinates)
                )
                print(f"Расстояние до {sub_id}: {distance:.2f} м")
                attenuation = self.attenuation / (1 + distance)
                noise = self.noise_level * np.random.randn(len(signal))
                delayed_signal = self._apply_propagation_delay(signal, distance)
                subscriber.receive(delayed_signal * attenuation + noise)
                print(f"Сигнал получен {sub_id} [Размер: {len(delayed_signal)}]")
        print(f"Передача завершена за {time.time() - start_time:.2f} сек")

    def _apply_propagation_delay(self, signal: np.ndarray, distance: float) -> np.ndarray:
        delay_samples = int((distance / 3e8) * 48000)
        return np.pad(signal, (delay_samples, 0), mode='constant')

    def transmit(self, signal: np.ndarray, tx_radio: 'DMRSubscriber'):
        """Передача сигнала в эфир с учетом расстояния"""
        for radio_id, radio in self.subscribers.items():
            if radio_id != tx_radio.subs_id:
                distance = np.linalg.norm(
                    np.array(tx_radio.coordinates) -
                    np.array(radio.coordinates)
                )
                attenuation = self.attenuation / (1 + distance)
                delayed_signal = self._apply_channel_effects(
                    signal * attenuation,
                    distance
                )
                radio.receive(delayed_signal)

    def _apply_channel_effects(self, signal: np.ndarray, distance: float, tx_radio: 'DMRSubscriber') -> np.ndarray:
        """Добавление эффектов распространения"""
        # Задержка сигнала (1 мс на 300 км)
        delay = distance / 3e8  # Задержка в секундах
        delay_samples = int(delay * tx_radio.sample_rate)
        signal = np.pad(signal, (delay_samples, 0))

        # Добавление шума
        noise = self.noise_floor * np.random.randn(len(signal))
        return signal + noise


class DMRMode(Enum):
    REPEATER = 1
    DIRECT = 2

class DMRSubscriber(Subscriber):
    FREQ_RANGE = (400e6, 430e6)  # 400-430 МГц
    SYMBOL_RATE = 4800
    DEVIATION = 1.8e3
    TDMA_SLOTS = 2
    DIRECT_SLOTS = 1
    FRAME_DURATION = 0.06

    def __init__(self,
                 subs_id: str,
                 channel: DMRChannel,
                 mode: DMRMode,
                 distance,
                 angle,
                 power: float=1.0,
                 slot=1):
        super().__init__(distance, angle, power)
        self.subs_id = subs_id
        # self.frequency = self._generate_frequency()
        self.frequency = 446.00625e6
        self.slot = slot
        self.mode = mode
        self.channel = channel
        self.signal_type = SignalType.DMR
        self._init_dmr_params()

        self.tx_thread = threading.Thread(target=self._transmission_loop)
        self.rx_thread = threading.Thread(target=self._reception_loop)

        self.tx_thread.daemon = True
        self.rx_thread.daemon = True
        self.tx_thread.start()
        self.rx_thread.start()
        self.modulation = "4-FSK"  # Тип модуляции
        self.color = "#2ca02c"  # Цвет для визуализации

    def _init_dmr_params(self):
        self.symbol_map = np.array([-1.8e3, -0.6e3, 0.6e3, 1.8e3])
        self.sync_bits = {
            DMRMode.DIRECT : np.array([1,0,1,0,1,1,0,0,1,1,1,0,0,1,0,1]),
            DMRMode.REPEATER : {
                1: np.array([1,1,1,0,1,0,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,0,1,0]),
                2: np.array([1,1,0,1,1,1,1,0,1,0,1,0,1,1,1,1,0,1,1,1,0,1,1,0])
            }
        }

    def push_to_talk(self):
        """Нажатие PTT для передачи"""
        print(f"\n[{self.subs_id}] PTT НАЖАТ")
        self._ptt = True

    def release_ptt(self):
        print(f"\n[{self.subs_id}] PTT ОТПУЩЕН")
        self._ptt = False

    def _transmission_loop(self):
        while True:
            if self._ptt:
                frame = self._generate_frame()
                self.channel.broadcast(frame, self)
                time.sleep(self.FRAME_DURATION)
            else:
                time.sleep(0.01)

    def _generate_frame(self) -> np.ndarray:
        if self.mode == DMRMode.DIRECT:
            return self._generate_direct_mode_frame()
        else:
            return self._generate_tdma_frame()

    def _generate_direct_mode_frame(self) -> np.ndarray:
        voice = self._generate_voice_data()
        crc = self._calculate_crc(voice)
        frame = np.concatenate([self.sync_bits[DMRMode.DIRECT], voice, crc])
        return self._modulate(frame)

    def _generate_tdma_frame(self) -> np.ndarray:
        sync = self.sync_bits[DMRMode.REPEATER][self.slot]
        voice = self._generate_voice_data()
        control = self._generate_control_data()
        frame = np.concatenate([sync, voice, control])
        return self._modulate(frame)

    def _modulate(self, bits: np.ndarray) -> np.ndarray:
        symbols = self._bits_to_symbols(bits)
        t = np.arange(len(symbols)) / self.SYMBOL_RATE
        freq = self.symbol_map[symbols]
        phase = 2 * np.pi * np.cumsum(freq) * t
        return np.exp(1j * phase)

    def _bits_to_symbols(self, bits: np.ndarray) -> np.ndarray:
        padded = np.pad(bits, (0, len(bits) % 2), 'constant')
        return 2 * padded[::2] + padded[1::2]

    def receive(self, signal: np.ndarray):
        print(f"{self.subs_id}: Получено {len(signal)} отсчетов")
        self.rx_buffer.put(signal)

    def _reception_loop(self):
        print(f"{self.subs_id}: Цикл приема запущен")
        while True:
            if not self.rx_buffer.empty():
                signal = self.rx_buffer.get()
                print(f"{self.subs_id}: Обработка сигнала [Размер: {len(signal)}]")
                self._process_signal(signal)

    def _process_signal(self, signal: np.ndarray):
        bits = self._demodulate(signal)
        if self._check_crc(bits):
            audio = self._decode_audio(bits)
            self._play_audio(audio)

    def _demodulate(self, signal: np.ndarray) -> np.ndarray:
        # Упрощенная демодуляция 4-FSK
        phase_diff = np.diff(np.angle(signal))
        symbols = np.digitize(phase_diff, [-np.pi / 2, 0, np.pi / 2])
        return self._symbols_to_bits(symbols)

    def _symbols_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        bits = np.zeros(2 * len(symbols), dtype=int)
        bits[::2] = symbols // 2
        bits[1::2] = symbols % 2
        return bits

    def _generate_voice_data(self) -> np.ndarray:
        # Имитация голосовых данных
        return np.random.randint(0, 2, 108)

    def _calculate_crc(self, data: np.ndarray) -> np.ndarray:
        # Заглушка для CRC
        return np.random.randint(0, 2, 16)

    def _check_crc(self, data: np.ndarray) -> bool:
        # Всегда возвращает True для примера
        return True

    def _decode_audio(self, bits: np.ndarray) -> np.ndarray:
        # Имитация декодирования аудио
        return np.random.rand(160)  # 20 ms audio frame

    def _play_audio(self, audio: np.ndarray):
        # Заглушка для воспроизведения аудио
        pass

class TETRASubscriber(Subscriber):
    FREQ_RANGE = (380e6, 400e6)  # 380-400 МГц
    def __init__(self, distance, angle, power=1.0):
        super().__init__(distance, angle, power)
        self.frequency = self._generate_frequency()
        self.signal_type = SignalType.TETRA
        self.modulation = "π/4-DQPSK"  # Тип модуляции
        self.timeslots = 4  # Число временных слотов
        self.color = "#f00"  # Цвет для визуализации

    def _generate_frequency(self) -> float:
        return np.random.uniform(*self.FREQ_RANGE)

    def generate_signal(self, num_samples: int, fs: float) -> np.ndarray:
        t = np.linspace(0, num_samples / fs, num_samples)
        phase = np.cumsum(np.random.choice([-1, 1], num_samples) * np.pi / 4)
        return 0.7 * np.exp(1j * phase) * self.power


class GSMSubscriber(Subscriber):
    FREQ_RANGE_900 = (890e6, 915e6) # 890-915 МГц
    FREQ_RANGE_1800 = (1800e6, 1900e6)
    def __init__(self, distance, angle, power=1.0):
        super.__init__(distance, angle, power)
        self.signal_type = SignalType.GSM
        self.modulation = "GMSK"

    def generate_signal(self, num_samples: int, fs: float) -> np.ndarray:
        pass
