"""FastAPI REST + WebSocket layer. Imports the engine; the engine never imports this."""

from elevator_mas.api.server import create_app

__all__ = ["create_app"]
