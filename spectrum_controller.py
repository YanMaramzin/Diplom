# spectrum_controller.py
import numpy as np
from signal_processing import calculate_spectrum, generate_dmr_signal

class SpectrumController:
    """Управляет отображением спектра."""
    def __init__(self, spectrum_view, sample_rate):
        self.spectrum_view = spectrum_view
        self.sample_rate = sample_rate

    def update_spectrum(self, subscribers):
        """Обновляет отображение спектра."""
        combined_signal = np.zeros(int(0.1 * self.sample_rate))
        num_samples = len(combined_signal)
        for sub in subscribers:
            signal = generate_dmr_signal(sub, num_samples, self.sample_rate)
            combined_signal += signal

        frequencies, spectrum_data = calculate_spectrum(combined_signal, self.sample_rate)
        self.spectrum_view.update_spectrum(spectrum_data, frequencies)