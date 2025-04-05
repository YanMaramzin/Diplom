import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

class DMRSubscriber:
    # Параметры сигнала
    SAMPLE_RATE = 48000  # Частота дискретизации
    SYMBOL_RATE = 4800  # Символьная скорость (символов/с)
    SYMBOL_DURATION = 1 / SYMBOL_RATE  # Длительность символа
    SAMPLE_PER_SYMBOL = int(SAMPLE_RATE * SYMBOL_DURATION)  # Отсчетов на символ
    num_symbols = 10  # Количество символов
    CARRIER_FREQ = 400e6
    DEVIATION = 648
    DMR_FREQUENCIES_MAP = {
        (0,0): 3 * DEVIATION,
        (0,1): DEVIATION,
        (1,0): -DEVIATION,
        (1,1): -3 * DEVIATION
    }

    def __init__(self):
        self.SYMBOL_DURATION = 1 / self.SYMBOL_RATE  # Длительность символа
        self.SAMPLE_PER_SYMBOL = int(self.SAMPLE_RATE * self.SYMBOL_DURATION)  # Отсчетов на символ
        self.frequencies = self._calculate_frequencies()

    def _calculate_frequencies(self):
        """Вычисляет частоты для каждого символа на основе несущей и девиации."""
        frequencies = [
            self.CARRIER_FREQ + self.DMR_FREQUENCIES_MAP[(0,0)],
            self.CARRIER_FREQ + self.DMR_FREQUENCIES_MAP[(0,1)],
            self.CARRIER_FREQ + self.DMR_FREQUENCIES_MAP[(1,0)],
            self.CARRIER_FREQ + self.DMR_FREQUENCIES_MAP[(1,1)]
        ]
        return frequencies

    def modulate(self, symbols):
        """Генерирует 4-FSK сигнал на основе символов."""
        t = np.linspace(0, self.SYMBOL_DURATION, self.SAMPLE_PER_SYMBOL, endpoint=False)
        signal = np.array([])
        for sym in symbols:
            freq = self.frequencies[sym]
            samples = np.sin(2 * np.pi * freq * t)
            signal = np.concatenate((signal, samples))
        return signal

    def add_noise(self, signal, snr_db):
        """Добавляет шум к сигналу с заданным SNR."""
        signal_power = np.mean(signal ** 2)
        noise_power_db = 10 * np.log10(signal_power) - snr_db
        noise_power = 10 ** (noise_power_db / 10)
        noise = np.sqrt(noise_power) * np.random.normal(0, 1, len(signal))
        noisy_signal = signal + noise
        return noisy_signal

    @staticmethod
    def generate_signal(bits):
        # Создание сигнала
        t = np.linspace(0, DMRSubscriber.SYMBOL_DURATION, DMRSubscriber.SAMPLE_PER_SYMBOL, endpoint=False)
        symbol = list(zip(bits[::2], bits[1::2]))
        signal = np.array([])
        for sym in symbol:
            freq = DMRSubscriber.CARRIER_FREQ + DMRSubscriber.DMR_FREQUENCIES_MAP[sym]
            print(freq)
            samples = np.sin(2 * np.pi * freq * t)
            signal = np.concatenate((signal, samples))

        return signal

    def demodulate(self, signal):
        # Демодуляция сигнала
        received_symbols = []
        num_symbols = len(signal) // self.SAMPLE_PER_SYMBOL
        for i in range(num_symbols):
            start = i * self.SAMPLE_PER_SYMBOL
            end = start + self.SAMPLE_PER_SYMBOL
            segment = signal[start:end]

            # Анализ спектра
            fft_result = np.fft.fft(segment)
            fft_magnitude = np.abs(fft_result)
            freqs = np.fft.fftfreq(self.SAMPLE_PER_SYMBOL, 1 / self.SAMPLE_RATE)

            # Поиск максимальной частоты
            max_idx = np.argmax(fft_magnitude[:self.SAMPLE_PER_SYMBOL // 2])
            detected_freq = freqs[max_idx]

            # Сопоставление с ближайшей частотой
            sym_idx = np.argmin(np.abs(detected_freq - np.array(self.frequencies)))
            received_symbols.append(sym_idx)
        return received_symbols

    def generate_symbols_from_bits(self, bit_string):
        """Генерирует символы DMR 4-FSK на основе битовой строки."""
        symbols = []
        for i in range(0, len(bit_string), 2):  # Берем по 2 бита
            try:
                bit_pair = bit_string[i:i+2] #  slice - обрабатывает короткую строку в конце
                symbol_key = bit_pair.zfill(2) # заполняем нулями, если меньше 2 бит
                symbol_key = (int(bit_pair[0]), int(bit_pair[1]))
                #  В DMR ключи битовых пар - строки
                symbol_index =  list(self.DMR_FREQUENCIES_MAP.keys()).index(symbol_key)
                symbols.append(symbol_index)
            except (KeyError, IndexError):
                print(f"Invalid bit pair: {bit_string[i:i+2]}.  Skipping.")
                continue # Или другая обработка ошибки.
        return np.array(symbols)


class DMRVisualizer:
    def __init__(self, subscriber):
        self.subscriber = subscriber

    def plot_signal(self, time_axis, signal, title, *subplot_num):
        """Отрисовывает временной график сигнала."""
        plt.subplot(*subplot_num)
        plt.plot(time_axis, signal)
        plt.title(title)
        plt.xlabel('Время (с)')
        plt.ylabel('Амплитуда')
        plt.grid(True)

    def plot_spectrum(self, signal, title, *subplot_num):
        """Отрисовывает спектр сигнала."""
        plt.subplot(*subplot_num)
        fft = np.fft.fft(signal)
        freqs = DMRSubscriber.CARRIER_FREQ + np.fft.fftfreq(len(signal), 1 / self.subscriber.SAMPLE_RATE)
        plt.plot(freqs[:len(freqs) // 2] / 1e6, np.abs(fft[:len(fft) // 2]))
        # for f in self.subscriber.frequencies:
        #     plt.axvline(f, color='r', linestyle='--', alpha=0.5)
        plt.title(title)
        plt.xlabel('Частота (МГц)')
        plt.ylabel('Амплитуда')
        plt.grid(True)

    def plot_demodulation_table(self, symbols, received_symbols, *subplot_num):
        """Отрисовывает таблицу результатов демодуляции."""
        plt.subplot(*subplot_num)
        plt.axis('off')
        num_symbols = len(symbols)
        table_data = [[f'Symbol {i}', f'Original: {symbols[i]}', f'Received: {received_symbols[i]}']
                      for i in range(num_symbols)]
        plt.table(cellText=table_data, colWidths=[0.2] * 3, loc='center')
        plt.title('Результаты демодуляции')

    def plot_constellation(self, signal: np.ndarray, *subplot_num):
        """Диаграмма созвездия"""
        plt.subplot(*subplot_num)
        # Выделение I/Q компонентов
        analytic_signal = signal[:len(signal)//2] + 1j*signal[len(signal)//2:]
        plt.scatter(np.real(analytic_signal), np.imag(analytic_signal), alpha=0.5)
        # plt.set_title('Диаграмма созвездия')
        plt.grid(True)

    def visualize(self, signal, noisy_signal, symbols, received_symbols, snr_db):
        """Визуализирует все данные."""
        plt.figure(figsize=(12, 10))

        # График исходного сигнала
        self.plot_signal(np.arange(len(signal)) / self.subscriber.SAMPLE_RATE, signal, 'Исходный сигнал', 5, 1, 1)

        # График зашумленного сигнала
        self.plot_signal(np.arange(len(noisy_signal)) / self.subscriber.SAMPLE_RATE, noisy_signal,
                         f'Зашумленный сигнал (SNR = {snr_db} dB)', 5, 1, 2)

        # Спектры
        self.plot_spectrum(signal, 'Спектр исходного сигнала', 5, 1, 3)
        self.plot_spectrum(noisy_signal, 'Спектр зашумленного сигнала', 5, 1, 4)
        self.plot_constellation(signal, 5, 1, 5)
        # Таблица демодулированных символов
        # self.plot_demodulation_table(symbols, received_symbols, 4, 1, 4)

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Создание экземпляра абонента DMR
    subscriber = DMRSubscriber()

    # Создание экземпляра визуализатора
    visualizer = DMRVisualizer(subscriber)

    # Параметры сигнала
    num_bits = 20  # Количество бит
    snr_db = 30  # SNR в dB

    # Генерация битовой последовательности и преобразование ее в символы DMR
    bit_string = ''.join(str(np.random.randint(0, 2)) for _ in range(num_bits))  # Генерация строки бит
    symbols = subscriber.generate_symbols_from_bits(bit_string)  # Преобразование в символы
    num_symbols = len(symbols)  # обновляем, если symbols изменились.

    # Модуляция сигнала
    signal = subscriber.modulate(symbols)

    # Добавление шума
    noisy_signal = subscriber.add_noise(signal, snr_db)

    # Демодуляция сигнала
    received_symbols = subscriber.demodulate(noisy_signal)

    # Визуализация результатов
    visualizer.visualize(signal, noisy_signal, symbols, received_symbols, snr_db)