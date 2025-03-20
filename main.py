from PyQt6.QtWidgets import QApplication
from signal_processor import SignalProcessor
from visualizer import Visualizer
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.signal import find_peaks
from enum import Enum
from collections import namedtuple
from scipy import signal

class App(QApplication):
    def __init__(self):
        super().__init__([])
        self.signal_processor = SignalProcessor()
        self.signal_processor.generate_subscribers()
        self.visualizer = Visualizer(self.signal_processor.subscribers)
        self.visualizer.show()

        # Связываем обновление данных
        self.visualizer.timer.timeout.connect(self.process_data)

    def process_data(self):
        signals = self.signal_processor.generate_signals()
        angles, spectrum = self.signal_processor.process_signals(signals)
        print("angles:", angles)
        freqs, spectrum_db = self.signal_processor.calculate_spectrum(signals)
        self.visualizer.update_plots(angles, spectrum, freqs, spectrum_db, self.signal_processor.subscribers)
        self.signal_processor.freq = self.visualizer.freq

if __name__ == "__main__":
    app = App()
    app.exec()