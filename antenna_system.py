class AntennaSystem:
    def __init__(self, antenna_positions):
        """
        Конструктор класса AntennaSystem.

        Args:
            antenna_positions (list of tuples):  Список позиций антенн в формате (x, y).
        """
        self.antenna_positions = antenna_positions #  Позиции антенн в 2D пространстве
        self.num_antennas = len(antenna_positions)

    def get_antenna_positions(self):
        """Возвращает позиции антенн."""
        return self.antenna_positions