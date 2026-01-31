"""Tests for event system."""

import json
import time
from datetime import datetime, timezone

import pytest

from vramcram.events.schema import AgentEvent
from vramcram.events.types import EventType


def test_agent_event_creation() -> None:
    """Test creating an AgentEvent."""
    event = AgentEvent(
        event_type=EventType.JOB_QUEUED,
        source_agent_id="coordinator-1",
        payload={"job_id": "123", "model": "test-model"},
    )

    assert event.event_type == EventType.JOB_QUEUED
    assert event.source_agent_id == "coordinator-1"
    assert event.payload["job_id"] == "123"
    assert event.correlation_id  # Auto-generated UUID


def test_agent_event_serialization() -> None:
    """Test event serialization to JSON."""
    event = AgentEvent(
        event_type=EventType.MODEL_LOADED,
        source_agent_id="manager-llama",
        payload={"model": "llama-3", "vram_mb": 4096},
        correlation_id="test-correlation-id",
    )

    json_str = event.to_json()
    data = json.loads(json_str)

    assert data["event_type"] == "model.loaded"
    assert data["source_agent_id"] == "manager-llama"
    assert data["payload"]["model"] == "llama-3"
    assert data["correlation_id"] == "test-correlation-id"
    assert "timestamp" in data


def test_agent_event_deserialization() -> None:
    """Test event deserialization from JSON."""
    json_str = json.dumps(
        {
            "event_type": "job.completed",
            "source_agent_id": "worker-1",
            "payload": {"job_id": "456", "duration": 5.2},
            "timestamp": "2024-01-15T10:30:00+00:00",
            "correlation_id": "abc-123",
        }
    )

    event = AgentEvent.from_json(json_str)

    assert event.event_type == EventType.JOB_COMPLETED
    assert event.source_agent_id == "worker-1"
    assert event.payload["job_id"] == "456"
    assert event.correlation_id == "abc-123"
    assert isinstance(event.timestamp, datetime)


def test_agent_event_roundtrip() -> None:
    """Test serialization/deserialization roundtrip."""
    original = AgentEvent(
        event_type=EventType.WORKER_HEARTBEAT,
        source_agent_id="worker-llama-1",
        payload={"status": "idle", "vram_mb": 4096},
    )

    json_str = original.to_json()
    restored = AgentEvent.from_json(json_str)

    assert restored.event_type == original.event_type
    assert restored.source_agent_id == original.source_agent_id
    assert restored.payload == original.payload
    assert restored.correlation_id == original.correlation_id


def test_agent_event_invalid_json() -> None:
    """Test deserialization with invalid JSON."""
    with pytest.raises(ValueError, match="Invalid event JSON"):
        AgentEvent.from_json("not valid json")


def test_agent_event_missing_fields() -> None:
    """Test deserialization with missing required fields."""
    json_str = json.dumps(
        {
            "event_type": "job.queued",
            # Missing source_agent_id
            "payload": {},
        }
    )

    with pytest.raises(ValueError, match="Invalid event JSON"):
        AgentEvent.from_json(json_str)


def test_agent_event_invalid_event_type() -> None:
    """Test deserialization with invalid event type."""
    json_str = json.dumps(
        {
            "event_type": "invalid.event.type",
            "source_agent_id": "test",
            "payload": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": "test",
        }
    )

    with pytest.raises(ValueError):
        AgentEvent.from_json(json_str)


def test_event_bus_publish_subscribe(redis_client):  # type: ignore[no-untyped-def]
    """Test basic publish/subscribe."""
    from vramcram.events.bus import EventBus

    bus = EventBus(redis_client)

    # Subscribe to event
    bus.subscribe(EventType.JOB_QUEUED)

    # Publish event
    event = AgentEvent(
        event_type=EventType.JOB_QUEUED,
        source_agent_id="test",
        payload={"job_id": "123"},
    )
    bus.publish(event)

    # Wait a moment for message to propagate
    time.sleep(0.1)

    # Receive event
    received_event = bus.wait_for_event(timeout=2.0)
    assert received_event is not None
    assert received_event.event_type == EventType.JOB_QUEUED
    assert received_event.payload["job_id"] == "123"

    bus.close()


def test_event_bus_pattern_subscription(redis_client):  # type: ignore[no-untyped-def]
    """Test pattern-based subscription."""
    from vramcram.events.bus import EventBus

    bus = EventBus(redis_client)

    # Subscribe to all job events
    bus.psubscribe("events:job.*")

    # Publish multiple job events
    events_to_publish = [
        EventType.JOB_QUEUED,
        EventType.JOB_DISPATCHED,
        EventType.JOB_COMPLETED,
    ]

    for event_type in events_to_publish:
        event = AgentEvent(
            event_type=event_type, source_agent_id="test", payload={"job_id": "123"}
        )
        bus.publish(event)

    time.sleep(0.1)

    # Receive all three events
    received_types = []
    for _ in range(3):
        event = bus.wait_for_event(timeout=2.0)
        if event:
            received_types.append(event.event_type)

    assert len(received_types) == 3
    assert EventType.JOB_QUEUED in received_types
    assert EventType.JOB_DISPATCHED in received_types
    assert EventType.JOB_COMPLETED in received_types

    bus.close()


def test_event_bus_wait_for_event_timeout(redis_client):  # type: ignore[no-untyped-def]
    """Test wait_for_event with timeout."""
    from vramcram.events.bus import EventBus

    bus = EventBus(redis_client)
    bus.subscribe(EventType.JOB_QUEUED)

    # Wait for event that never arrives
    event = bus.wait_for_event(timeout=0.5)
    assert event is None

    bus.close()


def test_event_bus_wait_for_specific_event(redis_client):  # type: ignore[no-untyped-def]
    """Test waiting for specific event type with predicate."""
    from vramcram.events.bus import EventBus

    bus = EventBus(redis_client)
    bus.psubscribe("events:job.*")

    # Publish several events
    bus.publish(
        AgentEvent(
            event_type=EventType.JOB_QUEUED,
            source_agent_id="test",
            payload={"job_id": "123"},
        )
    )
    bus.publish(
        AgentEvent(
            event_type=EventType.JOB_DISPATCHED,
            source_agent_id="test",
            payload={"job_id": "456"},
        )
    )
    bus.publish(
        AgentEvent(
            event_type=EventType.JOB_COMPLETED,
            source_agent_id="test",
            payload={"job_id": "789"},
        )
    )

    time.sleep(0.1)

    # Wait for specific event type
    event = bus.wait_for_event(event_type=EventType.JOB_COMPLETED, timeout=2.0)
    assert event is not None
    assert event.event_type == EventType.JOB_COMPLETED

    bus.close()


def test_event_bus_predicate_filter(redis_client):  # type: ignore[no-untyped-def]
    """Test event filtering with predicate function."""
    from vramcram.events.bus import EventBus

    bus = EventBus(redis_client)
    bus.subscribe(EventType.JOB_QUEUED)

    # Publish events with different job IDs
    bus.publish(
        AgentEvent(
            event_type=EventType.JOB_QUEUED,
            source_agent_id="test",
            payload={"job_id": "123"},
        )
    )
    bus.publish(
        AgentEvent(
            event_type=EventType.JOB_QUEUED,
            source_agent_id="test",
            payload={"job_id": "456"},
        )
    )

    time.sleep(0.1)

    # Wait for event with specific job_id
    event = bus.wait_for_event(
        predicate=lambda e: e.payload.get("job_id") == "456", timeout=2.0
    )
    assert event is not None
    assert event.payload["job_id"] == "456"

    bus.close()


# Pytest fixtures
@pytest.fixture
def redis_client():  # type: ignore[no-untyped-def]
    """Create a Redis client for testing."""
    import redis

    client = redis.Redis(host="localhost", port=6379, db=15, decode_responses=True)

    # Clean up before test
    client.flushdb()

    yield client

    # Clean up after test
    client.flushdb()
    client.close()
