from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

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


    @abstractmethod
    def generate_signal(self, num_samples: int, fs: float) -> np.ndarray:
        """Генерация сигнала абонента"""
        pass

    @property
    def coordinates(self) -> tuple:
        """Декартовы координаты в метрах"""
        x = self.distance * np.cos(np.radians(self.angle))
        y = self.distance * np.sin(np.radians(self.angle))
        return x, y

class DMRSubscriber(Subscriber):
    FREQ_RANGE = (400e6, 430e6)  # 400-430 МГц
    def __init__(self, distance, angle, power=1.0):
        super().__init__(distance, angle, power)
        self.frequency = self._generate_frequency()
        self.signal_type = SignalType.DMR
        self.modulation = "π/4-DQPSK"  # Тип модуляции
        self.timeslots = 4  # Число временных слотов
        self.color = "#2ca02c"  # Цвет для визуализации

    def generate_signal(self, num_samples: int, fs: float) -> np.ndarray:
        t = np.linspace(0, num_samples/fs, num_samples)
        phase = np.cumsum(np.random.choice([-1, 1], num_samples)*np.pi/4)
        return 0.7 * np.exp(1j*phase) * self.power

    def _generate_frequency(self) -> float:
        """Генерация частоты в заданном диапазоне"""
        return np.random.uniform(*self.FREQ_RANGE)


class TETRASubscriber(Subscriber):
    FREQ_RANGE = (380e6, 400e6)  # 380-400 МГц
    def __init__(self, distance, angle, power=1.0):
        super().__init__(distance, angle, power)
        self.frequency = self._generate_frequency()
        self.signal_type = SignalType.TETRA
        self.modulation = "π/4-DQPSK"  # Тип модуляции
        self.timeslots = 4  # Число временных слотов
        self.color = "#2ca02c"  # Цвет для визуализации

    def _generate_frequency(self) -> float:
        return np.random.uniform(*self.FREQ_RANGE)

    def generate_signal(self, num_samples: int, fs: float) -> np.ndarray:
        t = np.linspace(0, num_samples / fs, num_samples)
        phase = np.cumsum(np.random.choice([-1, 1], num_samples) * np.pi / 4)
        return 0.7 * np.exp(1j * phase) * self.power