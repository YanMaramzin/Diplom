import numpy as np
import subscribers
from scipy import signal

class SignalProcessor:
    def __init__(self, num_antennas=8, freq=900e6, range_limit=1000):
        self.num_antennas = num_antennas
        self.freq = freq
        self.wavelength = 3e8 / freq
        self.range_limit = range_limit
        self.subscribers = []
        self.sampling_rate = 2 * self.freq

        # Инициализация антенной решетки
        self.antenna_pos = self._create_antenna_array()

    def _create_antenna_array(self):
        angles = np.linspace(0, 2 * np.pi, self.num_antennas, endpoint=False)
        return np.column_stack([
            np.cos(angles),
            np.sin(angles)
        ]) * (self.wavelength / 2)

    def generate_subscribers(self, num_sources=5):
        """Генерация случайных абонентов"""
        self.subscribers = []
        for _ in range(num_sources):
            # Случайный выбор типа сигнала
            stype = np.random.choice([subscribers.SignalType.DMR, subscribers.SignalType.TETRA])

            # Параметры абонента
            distance = np.random.uniform(100, 1000)
            angle = np.random.uniform(0, 360)
            power = np.random.uniform(0.5, 1.0)

            # Создание объекта
            if stype == subscribers.SignalType.DMR:
                sub = subscribers.DMRSubscriber(distance, angle, power)
            else:
                sub = subscribers.TETRASubscriber(distance, angle, power)

            self.subscribers.append(sub)

    def generate_signals(self, num_samples=1024, fs=2e6):
        """Генерация сигналов для всех антенн"""
        signals = np.zeros((self.num_antennas, num_samples), dtype=np.complex64)
        for sub in self.subscribers:
            # Генерация сигнала абонента
            src_signal = sub.generate_signal(num_samples, fs)

            # Применение пространственных задержек
            delay_phase = self._calc_delay_phase(sub.angle)
            signals += src_signal * delay_phase[:, np.newaxis]

        noise = 0.1 * (np.random.randn(*signals.shape) + 1j * np.random.randn(*signals.shape))

        return signals + noise

    def _calc_delay_phase(self, angle):
        """Расчет фазовых задержек для антенной решетки"""
        return np.exp(-2j * np.pi * self.freq *
            (self.antenna_pos[:,0] * np.cos(np.radians(angle)) +
             self.antenna_pos[:,1] * np.sin(np.radians(angle)))/3e8)

    def process_signals(self, signals):
        # MUSIC Algorithm
        corr_matrix = signals @ signals.conj().T / signals.shape[1]
        eigvals, eigvecs = np.linalg.eigh(corr_matrix)
        noise_subspace = eigvecs[:, :-len(self.subscribers)]

        # Расчет спектра
        theta_range = np.linspace(0, 360, 360)
        spectrum = np.zeros_like(theta_range)
        for i, theta in enumerate(theta_range):
            a = np.exp(-2j * np.pi * self.freq *
                       (self.antenna_pos[:, 0] * np.cos(np.radians(theta)) +
                        self.antenna_pos[:, 1] * np.sin(np.radians(theta))) / 3e8)
            spectrum[i] = 1 / np.linalg.norm(noise_subspace.conj().T @ a) ** 2

        peaks = signal.find_peaks(spectrum, height=0.5 * np.max(spectrum))[0]
        return theta_range[peaks], spectrum

    def calculate_spectrum(self, signals):
        spectrum = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(signals.mean(0)))) + 1e-10)
        freqs = np.fft.fftshift(np.fft.fftfreq(signals.shape[1], 1 / 2e6)) / 1e6
        return freqs, spectrum