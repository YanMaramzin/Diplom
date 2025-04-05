import logging
import numpy as np
import matplotlib.pyplot as plt  # Убедитесь, что matplotlib импортируется только там, где он используется

from dmr_subscriber import DMRSubscriber
from communication_modes import DirectMode, RepeaterMode
from channel_model import AWGNChannel, FadingChannel
from observer import EventType
from tdma import TDMAFrame, Repeater
from visualization import DMRVisualizer
from simulation import MODE_SIMULATION_MAP, direct_mode_simulation, repeater_mode_simulation

# Настройка логгера
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================================
#  Основная логика примера использования
# ==============================================================================================
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

   #  Создание модели канала
    channel_model = AWGNChannel(snr_db=snr_db) #  AWGN Channel
    # channel_model = FadingChannel(snr_db=snr_db) # Fading Channel

    # ----------------------------------------------------------------------
    #  Настройка режима работы
    # ----------------------------------------------------------------------
    # Direct Mode
    communication_mode1 = DirectMode(channel=channel_model)
    communication_mode2 = DirectMode(channel=channel_model)

    # Repeater Mode
    # communication_mode1 = RepeaterMode(repeater, slot_number=0, channel=channel_model)
    # communication_mode2 = RepeaterMode(repeater, slot_number=1, channel=channel_model)

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
    signal2, received_signal, received_symbols1 = simulation_function(subscriber1, subscriber2, snr_db)  # Вызываем функцию
    # ==========================================================================

    # Вывод результатов для Subscriber1 (принятое сообщение)
    if received_symbols1 is not None:
        print(f"Subscriber {subscriber1.id} received symbols: {received_symbols1}")
        logging.info(f"Subscriber {subscriber1.id} received symbols: {received_symbols1}")
    else:
        print(f"Subscriber {subscriber1.id}: No message received.")
        logging.info(f"Subscriber {subscriber1.id}: No message received.")

    # Визуализация (только для subscriber1 для примера)
    #  Визуализация будет происходить в обработчиках событий