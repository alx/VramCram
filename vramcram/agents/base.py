"""Base agent class for all VramCram agents."""

import signal
import uuid
from abc import ABC, abstractmethod
from typing import Any

import structlog

from vramcram.events.bus import EventBus
from vramcram.events.schema import AgentEvent
from vramcram.events.types import EventType

logger = structlog.get_logger()


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Provides common functionality for event publishing, subscribing,
    and lifecycle management. All agents must implement the run() method.
    """

    def __init__(self, agent_id: str, event_bus: EventBus) -> None:
        """Initialize base agent.

        Args:
            agent_id: Unique identifier for this agent instance.
            event_bus: Event bus for communication.
        """
        self.agent_id = agent_id
        self.event_bus = event_bus
        self.logger = logger.bind(agent_id=agent_id)
        self._running = False
        self._shutdown_requested = False

    def publish(
        self,
        event_type: EventType,
        payload: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Publish an event to the event bus.

        Args:
            event_type: Type of event to publish.
            payload: Optional event payload.
            correlation_id: Optional correlation ID for request tracking.
        """
        if payload is None:
            payload = {}

        event = AgentEvent(
            event_type=event_type,
            source_agent_id=self.agent_id,
            payload=payload,
            correlation_id=correlation_id or str(uuid.uuid4()),
        )

        self.logger.debug(
            "publishing_event",
            event_type=event_type.value,
            correlation_id=event.correlation_id,
        )

        self.event_bus.publish(event)

    def subscribe(self, *event_types: EventType) -> None:
        """Subscribe to specific event types.

        Args:
            *event_types: Event types to subscribe to.
        """
        self.event_bus.subscribe(*event_types)
        self.logger.info(
            "subscribed_to_events",
            event_types=[et.value for et in event_types],
        )

    def psubscribe(self, *patterns: str) -> None:
        """Subscribe to event patterns.

        Args:
            *patterns: Channel patterns to subscribe to.
        """
        self.event_bus.psubscribe(*patterns)
        self.logger.info("subscribed_to_patterns", patterns=list(patterns))

    @abstractmethod
    def run(self) -> None:
        """Run the agent's main loop.

        This method must be implemented by subclasses.
        It should respect self._shutdown_requested for graceful shutdown.
        """
        pass

    def start(self) -> None:
        """Start the agent.

        Sets up signal handlers and publishes agent.started event.
        """
        self._running = True
        self._shutdown_requested = False

        # Set up signal handlers for graceful shutdown (only in main thread)
        try:
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)
        except ValueError:
            # Signal handlers only work in main thread - this is fine for testing
            self.logger.debug("signal_handlers_not_available")

        self.logger.info("agent_starting")
        self.publish(EventType.AGENT_STARTED)

        try:
            self.run()
        except Exception:
            self.logger.exception("agent_error")
            self.publish(EventType.AGENT_FAILED)
            raise
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the agent gracefully.

        Publishes agent.shutdown event and cleans up resources.
        """
        if self._running:
            self.logger.info("agent_stopping")
            self.publish(EventType.AGENT_SHUTDOWN)
            self._running = False
            self._shutdown_requested = True

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals.

        Args:
            signum: Signal number.
            frame: Current stack frame.
        """
        self.logger.info("received_shutdown_signal", signal=signum)
        self._shutdown_requested = True
