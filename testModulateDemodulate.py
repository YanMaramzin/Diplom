import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class CommunicationModeEnum(Enum):
    DIRECT = "direct"
    REPEATER = "repeater"

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

    def __init__(self, id):
        self.id = id
        self.SYMBOL_DURATION = 1 / self.SYMBOL_RATE  # Длительность символа
        self.SAMPLE_PER_SYMBOL = int(self.SAMPLE_RATE * self.SYMBOL_DURATION)  # Отсчетов на символ
        self.frequencies = self._calculate_frequencies()
        self.tx_buffer = [] # Буфер для передачи
        self.rx_buffer = [] # Буфер для приема
        self.communication_mode = None  # Объект, реализующий логику передачи

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

    def prepare_message(self, bit_string):
        """Готовит сообщение для передачи."""
        symbols = self.generate_symbols_from_bits(bit_string)
        signal = self.modulate(symbols)
        self.tx_buffer.append(signal)

    def transmit(self):
        """Передает сигнал из буфера."""
        if self.tx_buffer:
            return self.tx_buffer.pop(0)  # Передаем первый сигнал из буфера
        else:
            return None

    def receive(self, signal):
        """Принимает сигнал и демодулирует его."""
        if signal is not None:
            self.rx_buffer.append(signal)
            symbols = self.demodulate(signal)
            return symbols
        else:
            return None

    def set_communication_mode(self, communication_mode):
        """Устанавливает объект, реализующий логику связи."""
        self.communication_mode = communication_mode

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

class TDMAFrame:
    """Класс, представляющий TDMA кадр."""
    def __init__(self, num_slots, slot_duration):
        self.num_slots = num_slots
        self.slot_duration = slot_duration # В секундах
        self.slots = [None] * num_slots # None = слот свободен

    def insert_signal(self, signal, slot_number):
        """Вставляет сигнал в указанный слот."""
        if 0 <= slot_number < self.num_slots:
             self.slots[slot_number] = signal
        else:
            raise ValueError("Invalid slot number.")

    def get_slot_signal(self, slot_number):
        """Возвращает сигнал из указанного слота."""
        if 0 <= slot_number < self.num_slots:
            return self.slots[slot_number]
        else:
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
      return frame_signal

class CommunicationMode:
    """Абстрактный базовый класс для режимов связи."""
    def __init__(self, mode: CommunicationModeEnum):
        self.mode = mode

    def transmit(self, subscriber, destination_subscriber):
        """Передает сигнал от абонента (абстрактный метод)."""
        raise NotImplementedError

    def receive(self, signal, subscriber):
        """Принимает сигнал абонентом (абстрактный метод)."""
        raise NotImplementedError

class DirectMode(CommunicationMode):
    """Режим прямой связи."""
    def __init__(self):
        super().__init__(CommunicationModeEnum.DIRECT)

    def transmit(self, subscriber, destination_subscriber):
        """Передает сигнал напрямую другому абоненту."""
        return subscriber.transmit()

    def receive(self, signal, subscriber):
        """Принимает сигнал напрямую от другого абонента."""
        return subscriber.receive(signal)


class RepeaterMode(CommunicationMode):
    """Режим связи через ретранслятор."""
    def __init__(self, repeater, slot_number):
        super().__init__(CommunicationModeEnum.REPEATER)
        self.repeater = repeater
        self.slot_number = slot_number

    def transmit(self, subscriber):
        """Передает сигнал ретранслятору."""
        signal = subscriber.transmit()
        if signal is not None:
            return signal, self.slot_number  # Возвращаем и сигнал, и номер слота
        else:
            return None, None

    def receive(self, signal, subscriber):
        """Принимает сигнал от ретранслятора."""
        return subscriber.receive(signal)


class Repeater:
    """Класс, представляющий ретранслятор."""
    def __init__(self, num_slots, slot_duration):
        self.tdma_frame = TDMAFrame(num_slots, slot_duration)

    def receive(self, signal, slot_number):
        """Принимает сигнал от абонента и помещает его в TDMA кадр."""
        self.tdma_frame.insert_signal(signal, slot_number)

    def transmit(self):
        """Передает TDMA кадр."""
        return self.tdma_frame.get_frame_signal()


# ==============================================================================================
#  Словари режимов и визуализации
# ==============================================================================================
def direct_mode_simulation(subscriber1, subscriber2, snr_db):
    """Симуляция в прямом режиме."""
    signal1 = subscriber1.communication_mode.transmit(subscriber1, subscriber2)  # передаем
    signal2 = subscriber2.communication_mode.transmit(subscriber2, subscriber1)  # передаем

    # Subscriber1 передает, Subscriber2 принимает
    if signal1 is not None:
        noisy_signal1 = subscriber1.add_noise(signal1, snr_db)
        received_symbols2 = subscriber2.communication_mode.receive(noisy_signal1, subscriber2)  # принимаем
    else:
        received_symbols2 = None

    # Subscriber2 передает, Subscriber1 принимает
    if signal2 is not None:
        noisy_signal2 = subscriber2.add_noise(signal2, snr_db)
        received_symbols1 = subscriber1.communication_mode.receive(noisy_signal2, subscriber1)  # принимаем
    else:
        received_symbols1 = None

    return signal2, noisy_signal2, received_symbols1 # Передаем необходимые переменные для визуализации


def repeater_mode_simulation(subscriber1, subscriber2, repeater, snr_db):
    """Симуляция в режиме ретранслятора."""
    # Симуляция TDMA кадра (с ретранслятором)
    # Subscriber1 передает в первом слоте
    signal1, slot_number1 = subscriber1.communication_mode.transmit(subscriber1) # Передаем с указанием слота
    if signal1 is not None:
        noisy_signal1 = subscriber1.add_noise(signal1, snr_db) # Добавляем шум
        repeater.receive(noisy_signal1, slot_number1) # Ретранслятор принимает

    # Subscriber2 передает во втором слоте
    signal2, slot_number2 = subscriber2.communication_mode.transmit(subscriber2)  # Передаем с указанием слота
    if signal2 is not None:
        noisy_signal2 = subscriber2.add_noise(signal2, snr_db) # Добавляем шум
        repeater.receive(noisy_signal2, slot_number2)  # Ретранслятор принимает

    # Ретранслятор передает TDMA кадр
    tdma_frame_signal = repeater.transmit()

    # Subscriber2 принимает TDMA кадр и демодулирует (slot 1)
    received_signal_slot1 = repeater.tdma_frame.get_slot_signal(0)
    if received_signal_slot1 is not None:
        received_symbols2 = subscriber2.communication_mode.receive(received_signal_slot1, subscriber2)  # Принимаем
    else:
        received_symbols2 = None

    # Subscriber1 принимает TDMA кадр и демодулирует (slot 2)
    received_signal_slot2 = repeater.tdma_frame.get_slot_signal(1)
    if received_signal_slot2 is not None:
        received_symbols1 = subscriber1.communication_mode.receive(received_signal_slot2, subscriber1) # Принимаем
    else:
        received_symbols1 = None
    return signal2, received_signal_slot2, received_symbols1  # Передаем необходимые переменные для визуализации

#  Связываем режимы с функциями симуляции
MODE_SIMULATION_MAP = {
    CommunicationModeEnum.DIRECT: direct_mode_simulation,
    CommunicationModeEnum.REPEATER: repeater_mode_simulation
}

#  Связываем режимы с текстовым описанием для вывода
MODE_DISPLAY_MAP = {
    CommunicationModeEnum.DIRECT: "Direct Mode",
    CommunicationModeEnum.REPEATER: "Repeater Mode"
}

# if __name__ == '__main__':
#     # Создание экземпляра абонента DMR
#     subscriber1 = DMRSubscriber(id=1)
#     subscriber2 = DMRSubscriber(id=2)
#
#     # Создание экземпляра визуализатора
#     visualizer = DMRVisualizer(subscriber1)
#
#     # Параметры сигнала
#     num_bits = 20  # Количество бит
#     snr_db = 30  # SNR в dB
#
#     # Генерация битовой последовательности и преобразование ее в символы DMR
#     bit_string = ''.join(str(np.random.randint(0, 2)) for _ in range(num_bits))  # Генерация строки бит
#     symbols = subscriber1.generate_symbols_from_bits(bit_string)  # Преобразование в символы
#     num_symbols = len(symbols)  # обновляем, если symbols изменились.
#
#     # Модуляция сигнала
#     signal = subscriber1.modulate(symbols)
#
#     # Добавление шума
#     noisy_signal = subscriber1.add_noise(signal, snr_db)
#
#     # Демодуляция сигнала
#     received_symbols = subscriber1.demodulate(noisy_signal)
#
#     # Визуализация результатов
#     visualizer.visualize(signal, noisy_signal, symbols, received_symbols, snr_db)

if __name__ == '__main__':
    # Параметры TDMA
    num_slots = 2
    slot_duration = 0.05  # секунды

    # Создание абонентов DMR
    subscriber1 = DMRSubscriber(id=1)
    subscriber2 = DMRSubscriber(id=2)

    # Создание визуализатора (для subscriber1)
    visualizer = DMRVisualizer(subscriber1)

    # Создание ретранслятора
    repeater = Repeater(num_slots=num_slots, slot_duration=slot_duration)

    # Параметры сигнала
    num_bits = 20
    snr_db = 10

    # ----------------------------------------------------------------------
    #  Настройка режима работы
    # ----------------------------------------------------------------------
    # Direct Mode
    # communication_mode1 = DirectMode()
    # communication_mode2 = DirectMode()

    # Repeater Mode
    communication_mode1 = RepeaterMode(repeater, slot_number=0)
    communication_mode2 = RepeaterMode(repeater, slot_number=1)

    subscriber1.set_communication_mode(communication_mode1)
    subscriber2.set_communication_mode(communication_mode2)
    # ----------------------------------------------------------------------

    # 1. Subscriber1 готовит сообщение
    bit_string1 = ''.join(str(np.random.randint(0, 2)) for _ in range(num_bits))
    subscriber1.prepare_message(bit_string1)

    # 2. Subscriber2 готовит сообщение
    bit_string2 = ''.join(str(np.random.randint(0, 2)) for _ in range(num_bits))
    subscriber2.prepare_message(bit_string2)

    # ==========================================================================
    #  Симуляция передачи (без if/else)
    # ==========================================================================
    simulation_function = MODE_SIMULATION_MAP[subscriber1.communication_mode.mode]  # Получаем функцию симуляции
    signal2, received_signal, received_symbols1 = simulation_function(subscriber1, subscriber2, repeater, snr_db)  # Вызываем функцию
    # ==========================================================================

    # Вывод результатов для Subscriber1 (принятое сообщение)
    if received_symbols1 is not None:
        print(f"Subscriber {subscriber1.id} received symbols: {received_symbols1}")
    else:
        print(f"Subscriber {subscriber1.id}: No message received.")

    # Визуализация (только для subscriber1 для примера)
    if received_signal is not None and signal2 is not None:
        # Ограничиваем noisy_signal длиной signal для визуализации
        min_len = min(len(signal2), len(received_signal))
        signal2 = signal2[:min_len]
        received_signal = received_signal[:min_len]
        visualizer.visualize(signal2, received_signal, subscriber2.generate_symbols_from_bits(bit_string2), received_symbols1, snr_db)
    else:
        print(f"No signal to visualize for Subscriber 1 in {MODE_DISPLAY_MAP[subscriber1.communication_mode.mode]}")