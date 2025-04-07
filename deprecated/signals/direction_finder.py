import numpy as np
from scipy.signal import correlate
from typing import List, Dict
from subscribers import DMRSubscriber, DMRChannel

class MultiChannelDirectionFinder:
    pass

class DirectionFinder:
    def __init__(self, channel: DMRChannel, receivers: List[DMRSubscriber]):
        self.channel = channel
        self.receivers = receivers
        self.positions = np.array([r.coordinates for r in receivers])
        print(f"Пеленгатор инициализирован с {len(receivers)} приёмниками")

    def calculate_angle(self, signal_source: DMRSubscriber):
        # Измерение времени прихода сигнала
        delays = []
        for rx in self.receivers:
            # Получение сигнала из буфера
            if not rx.rx_buffer.empty():
                signal = rx.rx_buffer.get()
                # Поиск задержки относительно первого приёмника
                correlation = correlate(signal, self.reference_signal, mode='full')
                delay = np.argmax(correlation) - len(self.reference_signal) + 1
                delays.append(delay / rx.SAMPLE_RATE)

        # Триангуляция для определения направления
        if len(delays) >= 2:
            return self._triangulate(delays)
        return None

    def _triangulate(self, delays: List[float]):
        # Расчет угла с использованием разницы времён прихода (TDOA)
        # Реализация упрощённого алгоритма пеленгации
        delta_t = delays[1] - delays[0]
        distance = np.linalg.norm(self.positions[1] - self.positions[0])
        sin_theta = (delta_t * 343) / distance  # 343 m/s - скорость звука (пример)
        return np.degrees(np.arcsin(sin_theta))

    @property
    def reference_signal(self):
        # Опорный сигнал для корреляции (пример реализации)
        t = np.linspace(0, 0.1, 4800)
        return np.sin(2 * np.pi * 1000 * t)