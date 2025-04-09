import numpy as np
import logging

# Настройка логгера
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Установите желаемый уровень логирования

def generate_noise_spectrum(frequencies, noise_power):
    """Генерирует массив данных, представляющих спектр шума."""
    noise = np.random.normal(0, np.sqrt(noise_power), len(frequencies))
    return noise

def calculate_spectrum(signal, sample_rate):
    """Вычисляет спектр сигнала с использованием БПФ."""
    logger.debug("Начинаем вычисление спектра.")
    n = len(signal)
    logger.debug(f"Длина сигнала: {n}")

    try:
        yf = np.fft.fft(signal)
        xf = np.fft.fftfreq(n, 1 / sample_rate)
        logger.debug("БПФ успешно вычислен.")
        frequencies = xf[:n//2]
        amplitudes = np.abs(yf[:n//2])
        logger.debug(f"Количество частот: {len(frequencies)}, количество амплитуд: {len(amplitudes)}")
        return frequencies, amplitudes
    except Exception as e:
        logger.error(f"Ошибка при вычислении спектра: {e}")
        return None, None  # Возвращаем None в случае ошибки

def generate_dmr_signal(subscriber, num_samples, sample_rate):
    """Генерирует DMR-сигнал для абонента и добавляет шум."""
    logger.debug(f"Генерируем DMR сигнал для абонента {subscriber.id}")
    try:
        if subscriber.should_transmit():
            # Получаем последний сигнал из буфера (если есть)
            if subscriber.tx_buffer:
                signal = subscriber.tx_buffer[-1]
                logger.info("Сигнал получен из буфера.")
            else:
                logger.warning("Буфер передачи пуст. Генерируем только шум.")
                signal = np.zeros(num_samples)  # Создаем нулевой сигнал ПРАВИЛЬНОГО размера
        else:
            signal = np.zeros(num_samples)  # Абонент не должен передавать, генерируем тишину

        # Обрезаем или дополняем сигнал до нужной длины
        logger.debug(f"Длина сигнала до: {len(signal)}")
        if len(signal) > num_samples:
            signal = signal[:num_samples]
            logger.debug(f"Сигнал обрезан до {num_samples} отсчетов.")
        elif len(signal) < num_samples:
            signal = np.pad(signal, (0, num_samples - signal.size), 'constant')
            logger.debug(f"Сигнал дополнен до {num_samples} отсчетов.")

        logger.debug(f"Длина сигнала после: {len(signal)}")

        # Добавляем шум
        min_noise_power = 0.0001  # Минимальный уровень шума
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = max(min_noise_power, 0.001 * signal_power)  # Уровень шума
        noise = np.sqrt(noise_power / 2) * np.random.randn(num_samples)  # Генерируем шум

        #  Убеждаемся, что шум имеет ту же длину, что и сигнал
        if len(noise) > len(signal):
            noise = noise[:len(signal)]
        elif len(noise) < len(signal):
            noise = np.pad(noise, (0, signal.size - noise.size), 'constant')

        logger.debug(f"Размер сигнала перед сложением: {signal.shape}")
        logger.debug(f"Размер шума перед сложением: {noise.shape}")
        signal = signal + noise  # Смешиваем сигнал с шумом
        logger.debug("К сигналу добавлен шум.")
        return signal
    except Exception as e:
        logger.error(f"Ошибка при генерации DMR сигнала: {e}")
        # Генерируем шум ПРАВИЛЬНОГО размера
        noise = np.sqrt(0.1) * np.random.randn(num_samples)  # Просто шум
        logger.info(f"Только шум")
        return noise