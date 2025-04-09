import numpy as np
import logging

logger = logging.getLogger(__name__)

class DirectionFinder:
    def __init__(self, antenna_system, bearing_calculation_method="music"):
        """
        Конструктор класса DirectionFinder.

        Args:
            antenna_system (AntennaSystem): Объект, представляющий антенную систему.
            bearing_calculation_method (str): Метод расчета пеленга.
                                            Доступные методы: "music", "esprit".
                                            По умолчанию "music".
        """
        self.antenna_system = antenna_system
        self.bearing_calculation_method = bearing_calculation_method  # Метод расчета пеленга

    def set_bearing_calculation_method(self, method):
        """
        Устанавливает метод расчета пеленга.

        Args:
            method (str): Название метода ("music", "esprit").
        """
        if method in ("music", "esprit"):
            self.bearing_calculation_method = method
        else:
            logger.warning("Invalid bearing calculation method.  Using MUSIC.")
            self.bearing_calculation_method = "music"

    def estimate_bearing(self, spectrum, frequencies, carrier_frequency):
        """
        Оценивает пеленг на абонента.

        Args:
            spectrum (np.ndarray): Спектр сигнала.
            frequencies (np.ndarray): Массив частот, соответствующих спектру.
            carrier_frequency (float): Несущая частота сигнала.

        Returns:
            float: Оценка пеленга в градусах (относительно севера).  Возвращает None при ошибке.
        """
        try:
            #  Находим индекс частоты, наиболее близкой к несущей частоте
            index = np.argmin(np.abs(frequencies - carrier_frequency))

            #  Получаем сигнал на этой частоте
            received_signal = spectrum[index]

            #  Применяем выбранный метод пеленгования
            if self.bearing_calculation_method == "music":
                bearing = self._music_bearing(received_signal, carrier_frequency)
            elif self.bearing_calculation_method == "esprit":
                bearing = self._esprit_bearing(received_signal, carrier_frequency)
            else:
                logger.error("Invalid bearing calculation method.")
                return None

            return bearing
        except Exception as e:
            logger.error(f"Error estimating bearing: {e}")
            return None

    def _music_bearing(self, received_signal, carrier_frequency):
        """
        Оценивает пеленг на абонента с использованием MUSIC.
        Это заглушка, требующая реализации.

        Args:
            received_signal (complex): Принятый сигнал.
            carrier_frequency (float): Несущая частота сигнала.

        Returns:
            float: Оценка пеленга в градусах.  Возвращает None при ошибке.
        """
        #  TODO: Реализация MUSIC
        logger.warning("MUSIC bearing calculation not implemented.")
        return None

    def _esprit_bearing(self, received_signal, carrier_frequency):
        """
        Оценивает пеленг на абонента с использованием ESPRIT.
        Это заглушка, требующая реализации.

        Args:
            received_signal (complex): Принятый сигнал.
            carrier_frequency (float): Несущая частота сигнала.

        Returns:
            float: Оценка пеленга в градусах.  Возвращает None при ошибке.
        """
        #  TODO: Реализация ESPRIT
        logger.warning("ESPRIT bearing calculation not implemented.")
        return None