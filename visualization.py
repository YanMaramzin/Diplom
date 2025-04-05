import matplotlib.pyplot as plt
import logging
from observer import Observer, EventType
from dmr_subscriber import DMRSubscriber  # Import для SAMPLE_RATE
import numpy as np

class DMRVisualizer(Observer):
    def __init__(self, subscriber):
        self.subscriber = subscriber
        self.logger = logging.getLogger("DMRVisualizer")
        self.subscriber.attach(self)  # Подписываемся на события абонента

    def update(self, event_type: EventType, data):
        """Реагирует на события от DMRSubscriber."""
        if event_type == EventType.TRANSMIT:
            signal = data["signal"]
            self.logger.info("Received transmit event. Visualizing signal.")
            self.visualize_transmit(signal)
        elif event_type == EventType.RECEIVE:
            signal = data["signal"]
            symbols = data["symbols"]
            self.logger.info("Received receive event. Visualizing received signal.")
            self.visualize_receive(signal, symbols)
        elif event_type == EventType.DEMODULATE:
            symbols = data["symbols"]
            signal = data["signal"]
            self.logger.info("Received demodulate event. Visualizing demodulation.")
            self.visualize_demodulate(signal, symbols)
        else:
            self.logger.warning(f"Received unknown event type: {event_type}")

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

    def visualize_transmit(self, signal):
        """Визуализирует передаваемый сигнал."""
        plt.figure(figsize=(10, 6))
        self.plot_signal(np.arange(len(signal)) / self.subscriber.SAMPLE_RATE, signal, 'Передаваемый сигнал', 1, 1, 1)
        plt.tight_layout()
        plt.show()
        self.logger.info("Visualization transmit completed.")

    def visualize_receive(self, signal, symbols):
        """Визуализирует принимаемый сигнал и таблицу демодуляции."""
        plt.figure(figsize=(12, 6))
        # График принятого сигнала
        self.plot_signal(np.arange(len(signal)) / self.subscriber.SAMPLE_RATE, signal, 'Принятый сигнал', 2, 1, 1)
        # Таблица результатов демодуляции
        # self.plot_demodulation_table(subscriber2.generate_symbols_from_bits(bit_string2), symbols, 2, 1, 2)
        plt.tight_layout()
        plt.show()
        self.logger.info("Visualization receive completed.")

    def visualize_demodulate(self, signal, symbols):
        """Визуализирует результат демодуляции и спектр"""
        plt.figure(figsize=(12, 6))
        # График спектра
        self.plot_spectrum(signal, 'Спектр демодулированного сигнала', 2, 1, 1)
        # Таблица результатов демодуляции
        # self.plot_demodulation_table(self.subscriber.generate_symbols_from_bits(bit_string2), symbols, 2, 1, 2)
        plt.tight_layout()
        plt.show()
        self.logger.info("Visualization demodulate completed.")

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