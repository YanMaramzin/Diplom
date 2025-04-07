import numpy as np
import logging
from observer import Observable, EventType
from strategy import SimulationStrategy
import matplotlib.pyplot as plt

class DMRSubscriber(Observable):
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

    def __init__(self, id, latitude, longitude, simulation_strategy: SimulationStrategy, carrier_frequency, tdma):
        super().__init__()
        self.id = id
        self.SYMBOL_DURATION = 1 / self.SYMBOL_RATE  # Длительность символа
        self.SAMPLE_PER_SYMBOL = int(self.SAMPLE_RATE * self.SYMBOL_DURATION)  # Отсчетов на символ
        self.CARRIER_FREQ = carrier_frequency  # Несущая частота
        self.frequencies = self._calculate_frequencies()
        self.latitude = latitude # Координаты
        self.longitude = longitude
        self.tx_buffer = [] # Буфер для передачи
        self.rx_buffer = [] # Буфер для приема
        self.communication_mode = None  # Объект, реализующий логику передачи
        self.simulation_strategy = simulation_strategy  # Стратегия симуляции
        self.tdma = tdma  # Добавляем TDMA
        self.current_time = 0  # Текущее время симуляции
        self.logger = logging.getLogger(f"DMRSubscriber-{id}")  # Создаем логгер для каждого абонента
        self.data_generation_rate = 10  # Hz - пример
        self.num_bits = 200  # Длина сообщения

    def generate_data(self, num_samples):
        """Генерирует сообщение для передачи."""
        self.generate_random_message(self.num_bits)

    def simulate(self, subscriber2, repeater, snr_db):
        """Запускает симуляцию с использованием выбранной стратегии."""
        return self.simulation_strategy.simulate(self, subscriber2, snr_db)

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
            plt.plot(signal)
        self.logger.info(f"Modulated signal with {len(symbols)} symbols.")
        return signal

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
        self.logger.debug(f"Demodulated signal. Received symbols: {received_symbols}")
        self.notify(EventType.DEMODULATE, {"symbols": received_symbols, "signal": signal})  # Уведомляем наблюдателей
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

    def prepare_message(self, bit_string):
        """Готовит сообщение для передачи."""
        symbols = self.generate_symbols_from_bits(bit_string)
        signal = self.modulate(symbols)
        self.tx_buffer.append(signal)
        self.logger.info(f"Prepared message for transmission.  Bit string: {bit_string}")

    def transmit(self):
        """Передает сигнал из буфера."""
        if self.tx_buffer:
            signal = self.tx_buffer[-1]  # Возвращаем последний сигнал, не удаляя его
            self.logger.info("Returning signal for transmission.")
            self.notify(EventType.TRANSMIT, {"signal": signal})  # Уведомляем наблюдателей
            return signal
        else:
            self.logger.warning("No signal to transmit.")
            return None

    def receive(self, signal):
        """Принимает сигнал и демодулирует его."""
        if signal is not None:
            self.rx_buffer.append(signal)
            symbols = self.demodulate(signal)
            self.logger.info(f"Received signal. Demodulated symbols: {symbols}")
            self.notify(EventType.RECEIVE, {"signal": signal, "symbols": symbols})  # Уведомляем наблюдателей
            return symbols
        else:
            self.logger.warning("Received None signal.")
            return None

    def set_communication_mode(self, communication_mode):
        """Устанавливает объект, реализующий логику связи."""
        self.communication_mode = communication_mode
        self.logger.info(f"Communication mode set to: {communication_mode.mode.value}")

    def generate_random_message(self, num_bits):
        """Генерирует случайное сообщение и помещает его в буфер."""
        bit_string = ''.join(str(np.random.randint(0, 2)) for _ in range(num_bits))
        self.prepare_message(bit_string)
        self.logger.info(f"Сгенерировано случайное сообщение: {bit_string}")

    def get_current_frequencies(self):
      """Возвращает список текущих частот, на которых происходит излучение."""
      #  В простейшем случае - возвращаем все частоты
      return self.frequencies # Список частот

    def should_transmit(self):
        """Определяет, должен ли абонент передавать в данный момент времени."""
        should = self.tdma.is_active_slot(self.id, self.current_time)
        self.logger.debug(f"Subscriber {self.id} should_transmit: {should} at time {self.current_time}")
        return should

    def update(self, dt):
        """Обновляет состояние абонента (например, время)."""
        self.logger.debug(f"Updating subscriber {self.id} at time {self.current_time} with dt {dt}")
        self.current_time += dt
        if self.current_time % (1/self.data_generation_rate) < dt:
            self.generate_data(int(dt * 48000)) # 48000 - sample_rate

        self.logger.debug(f"Subscriber {self.id} current_time is now  {self.current_time} after update")