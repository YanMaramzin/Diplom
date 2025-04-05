import logging
from dmr_subscriber import DMRSubscriber
from communication_modes import CommunicationModeEnum

def direct_mode_simulation(subscriber1, subscriber2, snr_db):
    """Симуляция в прямом режиме."""
    logger = logging.getLogger("direct_mode_simulation")
    logger.info("Starting direct mode simulation.")

    signal1 = subscriber1.communication_mode.transmit(subscriber1, subscriber2)  # передаем
    signal2 = subscriber2.communication_mode.transmit(subscriber2, subscriber1)  # передаем

    # Subscriber1 передает, Subscriber2 принимает
    if signal1 is not None:
        received_symbols2 = subscriber2.communication_mode.receive(signal1, subscriber2)  # принимаем
    else:
        received_symbols2 = None

    # Subscriber2 передает, Subscriber1 принимает
    if signal2 is not None:
        received_symbols1 = subscriber1.communication_mode.receive(signal2, subscriber1)  # принимаем
    else:
        received_symbols1 = None

    logger.info("Direct mode simulation completed.")
    return signal2, signal2, received_symbols1 # Передаем необходимые переменные для визуализации

def repeater_mode_simulation(subscriber1, subscriber2, repeater, snr_db):
    """Симуляция в режиме ретранслятора."""
    logger = logging.getLogger("repeater_mode_simulation")
    logger.info("Starting repeater mode simulation.")

    # Симуляция TDMA кадра (с ретранслятором)
    # Subscriber1 передает в первом слоте
    signal1, slot_number1 = subscriber1.communication_mode.transmit(subscriber1) # Передаем с указанием слота
    if signal1 is not None:
        repeater.receive(signal1, slot_number1) # Ретранслятор принимает

    # Subscriber2 передает во втором слоте
    signal2, slot_number2 = subscriber2.communication_mode.transmit(subscriber2)  # Передаем с указанием слота
    if signal2 is not None:
        repeater.receive(signal2, slot_number2)  # Ретранслятор принимает

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

    logger.info("Repeater mode simulation completed.")
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
