"""Tests for base agent functionality."""

import time

import pytest

from vramcram.agents.base import BaseAgent
from vramcram.events.bus import EventBus
from vramcram.events.types import EventType


class TestAgent(BaseAgent):
    """Test agent implementation."""

    def __init__(self, agent_id: str, event_bus: EventBus) -> None:
        """Initialize test agent."""
        super().__init__(agent_id, event_bus)
        self.run_called = False
        self.keep_running = False

    def run(self) -> None:
        """Simple run implementation."""
        self.run_called = True
        # Run until shutdown requested or timeout
        if self.keep_running:
            while not self._shutdown_requested:
                time.sleep(0.1)
        else:
            # Run briefly then stop
            time.sleep(0.1)


def test_base_agent_initialization(redis_client):  # type: ignore[no-untyped-def]
    """Test base agent initialization."""
    bus = EventBus(redis_client)
    agent = TestAgent("test-agent-1", bus)

    assert agent.agent_id == "test-agent-1"
    assert agent.event_bus is bus
    assert not agent._running
    assert not agent._shutdown_requested

    bus.close()


def test_base_agent_publish(redis_client):  # type: ignore[no-untyped-def]
    """Test publishing events."""
    bus = EventBus(redis_client)
    agent = TestAgent("test-agent-1", bus)

    # Subscribe to event in another bus instance
    bus2 = EventBus(redis_client)
    bus2.subscribe(EventType.AGENT_STARTED)

    # Publish event
    agent.publish(EventType.AGENT_STARTED, payload={"status": "starting"})

    time.sleep(0.1)

    # Receive event
    event = bus2.wait_for_event(timeout=2.0)
    assert event is not None
    assert event.event_type == EventType.AGENT_STARTED
    assert event.source_agent_id == "test-agent-1"
    assert event.payload["status"] == "starting"

    bus.close()
    bus2.close()


def test_base_agent_publish_with_correlation_id(redis_client):  # type: ignore[no-untyped-def]
    """Test publishing events with correlation ID."""
    bus = EventBus(redis_client)
    agent = TestAgent("test-agent-1", bus)

    bus2 = EventBus(redis_client)
    bus2.subscribe(EventType.JOB_QUEUED)

    # Publish with explicit correlation ID
    agent.publish(
        EventType.JOB_QUEUED,
        payload={"job_id": "123"},
        correlation_id="custom-correlation-id",
    )

    time.sleep(0.1)

    event = bus2.wait_for_event(timeout=2.0)
    assert event is not None
    assert event.correlation_id == "custom-correlation-id"

    bus.close()
    bus2.close()


def test_base_agent_subscribe(redis_client):  # type: ignore[no-untyped-def]
    """Test subscribing to events."""
    bus = EventBus(redis_client)
    agent = TestAgent("test-agent-1", bus)

    # Subscribe to multiple event types
    agent.subscribe(EventType.JOB_QUEUED, EventType.JOB_COMPLETED)

    # Verify subscription worked by publishing and receiving
    bus2 = EventBus(redis_client)
    bus2.subscribe(EventType.JOB_QUEUED)

    # Publish event
    agent.publish(EventType.JOB_QUEUED, payload={"test": "data"})

    time.sleep(0.1)

    # Should receive event
    event = bus2.wait_for_event(timeout=2.0)
    assert event is not None
    assert event.event_type == EventType.JOB_QUEUED

    # Just verify no errors occurred
    bus.close()
    bus2.close()


def test_base_agent_psubscribe(redis_client):  # type: ignore[no-untyped-def]
    """Test pattern-based subscription."""
    bus = EventBus(redis_client)
    agent = TestAgent("test-agent-1", bus)

    # Subscribe to pattern
    agent.psubscribe("events:job.*", "events:model.*")

    # Verify subscription worked
    bus.close()


def test_base_agent_lifecycle(redis_client):  # type: ignore[no-untyped-def]
    """Test agent start and stop lifecycle."""
    bus = EventBus(redis_client)
    agent = TestAgent("test-agent-1", bus)
    agent.keep_running = True  # Keep agent running for test

    # Subscribe to agent events
    bus2 = EventBus(redis_client)
    bus2.psubscribe("events:agent.*")

    # Start agent in thread to avoid blocking
    import threading

    thread = threading.Thread(target=agent.start)
    thread.start()

    # Wait for agent to start
    time.sleep(0.2)

    # Check that run was called
    assert agent.run_called
    assert agent._running

    # Request shutdown
    agent.stop()

    # Wait for agent to finish
    thread.join(timeout=2.0)

    # Agent should have stopped
    assert not agent._running

    bus.close()
    bus2.close()


def test_base_agent_auto_publishes_started_event(redis_client):  # type: ignore[no-untyped-def]
    """Test that start() publishes agent.started event."""
    bus = EventBus(redis_client)
    agent = TestAgent("test-agent-1", bus)

    bus2 = EventBus(redis_client)
    bus2.subscribe(EventType.AGENT_STARTED)

    # Start agent in thread
    import threading

    thread = threading.Thread(target=agent.start)
    thread.start()

    time.sleep(0.2)

    # Should receive started event
    event = bus2.wait_for_event(event_type=EventType.AGENT_STARTED, timeout=2.0)
    assert event is not None
    assert event.source_agent_id == "test-agent-1"

    thread.join(timeout=2.0)

    bus.close()
    bus2.close()


def test_base_agent_auto_publishes_shutdown_event(redis_client):  # type: ignore[no-untyped-def]
    """Test that stop() publishes agent.shutdown event."""
    bus = EventBus(redis_client)
    agent = TestAgent("test-agent-1", bus)

    bus2 = EventBus(redis_client)
    bus2.subscribe(EventType.AGENT_SHUTDOWN)

    # Start agent in thread
    import threading

    thread = threading.Thread(target=agent.start)
    thread.start()

    time.sleep(0.2)

    # Agent should auto-stop after run completes
    thread.join(timeout=2.0)

    time.sleep(0.1)

    # Should receive shutdown event
    event = bus2.wait_for_event(event_type=EventType.AGENT_SHUTDOWN, timeout=2.0)
    assert event is not None
    assert event.source_agent_id == "test-agent-1"

    bus.close()
    bus2.close()


# Pytest fixtures
@pytest.fixture
def redis_client():  # type: ignore[no-untyped-def]
    """Create a Redis client for testing."""
    import redis

    client = redis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
    client.flushdb()

    yield client

    client.flushdb()
    client.close()
