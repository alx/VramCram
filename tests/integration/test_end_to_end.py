"""End-to-end integration tests for VramCram.

These tests require a running Redis instance and mock GPU hardware.
Run with: pytest tests/integration/ -v

Note: These tests are designed to be extended with full system integration
once GPU hardware and model files are available.
"""

import asyncio
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import redis

from vramcram.config.models import (
    APIConfig,
    LLMModelConfig,
    ModelsConfig,
    RedisConfig,
    VramCramConfig,
    VRAMConfig,
)
from vramcram.events.bus import EventBus
from vramcram.events.types import EventType
from vramcram.queue.job import Job, JobStatus
from vramcram.redis.client import RedisClientFactory


@pytest.fixture
def redis_config() -> RedisConfig:
    """Create test Redis configuration."""
    return RedisConfig(db=15)  # Use separate DB for testing


@pytest.fixture
def test_config() -> VramCramConfig:
    """Create test configuration."""
    return VramCramConfig(
        redis=RedisConfig(db=15),
        vram=VRAMConfig(total_mb=16384, safety_margin_mb=512),
        api=APIConfig(port=8001),
        models=ModelsConfig(
            llm=[
                LLMModelConfig(
                    name="test-llm",
                    model_path=Path("/fake/path/model.gguf"),
                    vram_mb=4096,
                )
            ]
        ),
    )


@pytest.fixture
def redis_factory(redis_config: RedisConfig) -> RedisClientFactory:
    """Create Redis factory for testing."""
    factory = RedisClientFactory(redis_config)
    yield factory

    # Cleanup
    client = factory.create_client()
    client.flushdb()
    factory.close()


@pytest.fixture
def event_bus(redis_factory: RedisClientFactory) -> EventBus:
    """Create event bus for testing."""
    client = redis_factory.create_client()
    return EventBus(client)


class TestJobLifecycle:
    """Test job lifecycle from submission to completion."""

    def test_job_submission_and_storage(
        self, redis_factory: RedisClientFactory, event_bus: EventBus
    ) -> None:
        """Test submitting a job and storing in Redis."""
        client = redis_factory.create_client()

        # Create job
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            model="test-model",
            prompt="test prompt",
            params={"max_tokens": 100},
        )

        # Store in Redis
        job_key = f"job:{job_id}"
        client.hset(job_key, mapping=job.to_dict())

        # Add to stream
        stream_id = client.xadd(
            "jobs:stream",
            {"job_id": job_id, "model": "test-model"},
        )

        assert stream_id is not None

        # Retrieve and verify
        stored_data = client.hgetall(job_key)
        assert stored_data["job_id"] == job_id
        assert stored_data["model"] == "test-model"

        # Verify stream
        stream_data = client.xrange("jobs:stream")
        assert len(stream_data) > 0

    def test_event_publishing_and_subscription(
        self, event_bus: EventBus
    ) -> None:
        """Test event publishing and subscription."""
        from vramcram.events.schema import AgentEvent

        # Subscribe to event
        event_bus.subscribe(EventType.JOB_QUEUED)

        # Publish event
        test_event = AgentEvent(
            event_type=EventType.JOB_QUEUED,
            source_agent_id="test_agent",
            payload={"job_id": "test-123", "model": "test-model"},
        )
        event_bus.publish(test_event)

        # Listen for event (with timeout)
        received = False
        event = event_bus.wait_for_event(timeout=2.0)
        if event and event.event_type == EventType.JOB_QUEUED:
            assert event.payload["job_id"] == "test-123"
            received = True

        assert received, "Event not received"

    def test_job_status_transitions(
        self, redis_factory: RedisClientFactory
    ) -> None:
        """Test job status transitions through lifecycle."""
        client = redis_factory.create_client()

        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            model="test-model",
            prompt="test prompt",
        )

        # Initial status: QUEUED
        assert job.status == JobStatus.QUEUED

        job_key = f"job:{job_id}"
        client.hset(job_key, mapping=job.to_dict())

        # Transition to DISPATCHED
        job.status = JobStatus.DISPATCHED
        client.hset(job_key, mapping=job.to_dict())

        # Transition to PROCESSING
        job.status = JobStatus.PROCESSING
        client.hset(job_key, mapping=job.to_dict())

        # Transition to COMPLETED
        job.status = JobStatus.COMPLETED
        job.result = '{"text": "test result"}'
        job.duration_ms = 1000
        client.hset(job_key, mapping=job.to_dict())

        # Verify final state
        final_data = client.hgetall(job_key)
        assert final_data["status"] == "completed"
        assert final_data["duration_ms"] == "1000"

    def test_consumer_group_creation(
        self, redis_factory: RedisClientFactory
    ) -> None:
        """Test creating consumer group for job stream."""
        client = redis_factory.create_client()

        stream_name = "jobs:stream"
        group_name = "workers-test-model"

        # Create stream with initial entry
        client.xadd(stream_name, {"init": "true"})

        # Create consumer group
        try:
            client.xgroup_create(
                name=stream_name,
                groupname=group_name,
                id="0",
                mkstream=True,
            )
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

        # Verify group exists
        groups = client.xinfo_groups(stream_name)
        group_names = [g["name"] for g in groups]
        assert group_name in group_names


class TestMockedSystemIntegration:
    """Test system integration with mocked components.

    These tests verify the integration between components without
    requiring actual GPU hardware or model files.
    """

    @patch("vramcram.gpu.vram_tracker.pynvml")
    def test_vram_tracking_integration(self, mock_pynvml: MagicMock) -> None:
        """Test VRAM tracker integration with coordinator."""
        from vramcram.gpu.vram_tracker import VRAMTracker

        # Mock NVML
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle

        mock_mem_info = MagicMock()
        mock_mem_info.total = 16 * 1024 * 1024 * 1024
        mock_mem_info.used = 4 * 1024 * 1024 * 1024
        mock_mem_info.free = 12 * 1024 * 1024 * 1024
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem_info

        # Create tracker
        tracker = VRAMTracker(gpu_index=0)

        # Test VRAM queries
        state = tracker.get_vram_state()
        assert state.free_mb == 12 * 1024

        # Test sufficiency check
        assert tracker.has_sufficient_vram(4 * 1024) is True
        assert tracker.has_sufficient_vram(16 * 1024) is False


@pytest.mark.skipif(
    not Path("/usr/bin/redis-server").exists(),
    reason="Redis server not available",
)
class TestRealRedisIntegration:
    """Tests that require real Redis instance.

    Skip these if Redis is not available.
    """

    def test_redis_connection(self, redis_factory: RedisClientFactory) -> None:
        """Test basic Redis connection."""
        client = redis_factory.create_client()

        # Test ping
        assert client.ping() is True

        # Test basic operations
        client.set("test_key", "test_value")
        assert client.get("test_key") == "test_value"

    def test_redis_streams(self, redis_factory: RedisClientFactory) -> None:
        """Test Redis Streams operations."""
        client = redis_factory.create_client()

        stream_name = "test:stream"

        # Add entries
        id1 = client.xadd(stream_name, {"data": "entry1"})
        id2 = client.xadd(stream_name, {"data": "entry2"})

        assert id1 is not None
        assert id2 is not None

        # Read entries
        entries = client.xrange(stream_name)
        assert len(entries) == 2

        # Test XREADGROUP
        client.xgroup_create(stream_name, "test_group", id="0", mkstream=True)

        messages = client.xreadgroup(
            groupname="test_group",
            consumername="test_consumer",
            streams={stream_name: ">"},
            count=1,
        )

        assert len(messages) > 0


# Note: Full end-to-end tests with actual model loading and inference
# would require GPU hardware, model files, and significantly more setup.
# These should be added once the development environment is fully configured.
