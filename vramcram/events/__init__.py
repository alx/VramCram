"""Event system for agent communication."""

from vramcram.events.bus import EventBus
from vramcram.events.schema import AgentEvent
from vramcram.events.types import EventType

__all__ = ["EventBus", "AgentEvent", "EventType"]
