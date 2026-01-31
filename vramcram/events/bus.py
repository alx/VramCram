"""Event bus implementation using Redis Pub/Sub."""

import time
from typing import Callable, Iterator

import redis

from vramcram.events.schema import AgentEvent
from vramcram.events.types import EventType


class EventBus:
    """Event bus for agent communication via Redis Pub/Sub.

    Provides publish/subscribe functionality with pattern matching support.
    Events are serialized as JSON and sent over Redis Pub/Sub channels.
    """

    def __init__(self, redis_client: redis.Redis) -> None:
        """Initialize event bus.

        Args:
            redis_client: Redis client instance.
        """
        self._redis = redis_client
        self._pubsub = self._redis.pubsub()

    def publish(self, event: AgentEvent) -> None:
        """Publish an event to the event bus.

        Args:
            event: Event to publish.
        """
        channel = f"events:{event.event_type.value}"
        self._redis.publish(channel, event.to_json())

    def subscribe(self, *event_types: EventType) -> None:
        """Subscribe to specific event types.

        Args:
            *event_types: Event types to subscribe to.
        """
        channels = [f"events:{event_type.value}" for event_type in event_types]
        self._pubsub.subscribe(*channels)

        # Consume subscription confirmation messages
        for _ in range(len(channels)):
            msg = self._pubsub.get_message(timeout=1.0)
            if msg and msg["type"] == "subscribe":
                continue

    def psubscribe(self, *patterns: str) -> None:
        """Subscribe to event patterns.

        Args:
            *patterns: Channel patterns to subscribe to (e.g., "events:job.*").
        """
        self._pubsub.psubscribe(*patterns)

        # Consume subscription confirmation messages
        for _ in range(len(patterns)):
            msg = self._pubsub.get_message(timeout=1.0)
            if msg and msg["type"] == "psubscribe":
                continue

    def listen(self) -> Iterator[AgentEvent]:
        """Listen for events on subscribed channels.

        Yields:
            AgentEvent instances as they arrive.

        Note:
            This is a blocking iterator. Use wait_for_event() for timeout support.
        """
        for message in self._pubsub.listen():
            if message["type"] in ("message", "pmessage"):
                try:
                    event = AgentEvent.from_json(message["data"])
                    yield event
                except ValueError:
                    # Skip invalid events
                    continue

    def wait_for_event(
        self,
        event_type: EventType | None = None,
        predicate: Callable[[AgentEvent], bool] | None = None,
        timeout: float = 30.0,
    ) -> AgentEvent | None:
        """Wait for a specific event with timeout.

        Args:
            event_type: Optional event type to filter for.
            predicate: Optional predicate function to filter events.
            timeout: Maximum time to wait in seconds.

        Returns:
            Matching event or None if timeout.
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            message = self._pubsub.get_message(timeout=1.0)

            if message and message["type"] in ("message", "pmessage"):
                try:
                    event = AgentEvent.from_json(message["data"])

                    # Check event type filter
                    if event_type and event.event_type != event_type:
                        continue

                    # Check predicate filter
                    if predicate and not predicate(event):
                        continue

                    return event
                except ValueError:
                    # Skip invalid events
                    continue

        return None

    def unsubscribe(self) -> None:
        """Unsubscribe from all channels."""
        self._pubsub.unsubscribe()
        self._pubsub.punsubscribe()

    def close(self) -> None:
        """Close the event bus and underlying connections."""
        self._pubsub.close()
