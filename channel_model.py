import numpy as np
import logging
from abc import ABC, abstractmethod
import commpy.channels as channels

class ChannelModel(ABC):
    """Абстрактный базовый класс для моделей канала."""
    def __init__(self, snr_db):
        self.snr_db = snr_db
        self.logger = logging.getLogger("ChannelModel")

    @abstractmethod
    def transmit(self, signal):
        """Передает сигнал через канал (абстрактный метод)."""
        raise NotImplementedError

class AWGNChannel(ChannelModel):
    """Канал с аддитивным белым гауссовским шумом (AWGN)."""
    def __init__(self, snr_db):
        super().__init__(snr_db)
        self.logger = logging.getLogger("AWGNChannel")

    def transmit(self, signal):
        """Добавляет AWGN шум к сигналу."""
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = signal_power / (10**(self.snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.normal(0, 1, size=signal.shape) + 1j * np.random.normal(0, 1, size=signal.shape))
        noisy_signal = signal + noise
        self.logger.debug(f"Transmitted signal through AWGN channel with SNR = {self.snr_db} dB.")
        return noisy_signal

class FadingChannel(ChannelModel):
    """Модель канала с замираниями Rayleigh."""
    def __init__(self, snr_db, max_Doppler_shift=10):
        super().__init__(snr_db)
        self.max_doppler_shift = max_Doppler_shift
        self.channel = channels.RayleighFadingChannel(fading_characteristic='flat',
                                              maximum_doppler_shift=self.max_doppler_shift,
                                              sampling_frequency=DMRSubscriber.SAMPLE_RATE,
                                              seed=None)
        self.logger = logging.getLogger("FadingChannel")

    def transmit(self, signal):
        """Передает сигнал через канал с замираниями."""
        #  На CommPy требуется, чтобы  signal был комплексным
        signal_complex = signal.astype(complex)
        faded_signal = self.channel.propagate(signal_complex)
        #  Преобразуем обратно в numpy.array
        faded_signal = np.array(faded_signal)
        signal_power = np.mean(np.abs(faded_signal)**2)
        noise_power = signal_power / (10**(self.snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.normal(0, 1, size=faded_signal.shape) + 1j * np.random.normal(0, 1, size=faded_signal.shape))
        noisy_signal = faded_signal + noise
        self.logger.debug(f"Transmitted signal through Fading channel with SNR = {self.snr_db} dB.")
        return noisy_signal