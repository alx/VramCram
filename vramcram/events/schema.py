"""Event schema definitions."""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from vramcram.events.types import EventType


@dataclass
class AgentEvent:
    """Event for agent-to-agent communication.

    All agent communication happens via events published to Redis Pub/Sub.
    Events carry metadata for tracing and correlation.
    """

    event_type: EventType
    source_agent_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_json(self) -> str:
        """Serialize event to JSON string.

        Returns:
            JSON string representation.
        """
        data = asdict(self)
        # Convert EventType enum to string
        data["event_type"] = self.event_type.value
        # Convert datetime to ISO format string
        data["timestamp"] = self.timestamp.isoformat()
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> "AgentEvent":
        """Deserialize event from JSON string.

        Args:
            json_str: JSON string representation.

        Returns:
            AgentEvent instance.

        Raises:
            ValueError: If JSON is invalid or missing required fields.
        """
        try:
            data = json.loads(json_str)

            # Convert string back to EventType enum
            data["event_type"] = EventType(data["event_type"])

            # Convert ISO format string back to datetime
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])

            return cls(**data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Invalid event JSON: {e}") from e

    def __str__(self) -> str:
        """String representation for logging."""
        return (
            f"AgentEvent(type={self.event_type.value}, "
            f"source={self.source_agent_id}, "
            f"correlation_id={self.correlation_id})"
        )
