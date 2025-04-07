# map_controller.py
class MapController:
    """Управляет отображением абонентов на карте."""
    def __init__(self, map_view):
        self.map_view = map_view

    def update_map(self, subscribers):
        """Обновляет отображение абонентов на карте."""
        self.map_view.set_subscribers(subscribers)