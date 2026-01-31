"""Unit tests for API Gateway."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from vramcram.api.gateway import create_app
from vramcram.config.models import (
    APIConfig,
    LLMModelConfig,
    ModelsConfig,
    RedisConfig,
    VramCramConfig,
    VRAMConfig,
)
from vramcram.events.bus import EventBus
from vramcram.queue.job import JobStatus
from vramcram.redis.client import RedisClientFactory


@pytest.fixture
def mock_config() -> VramCramConfig:
    """Create mock configuration."""
    return VramCramConfig(
        redis=RedisConfig(),
        vram=VRAMConfig(total_mb=16384, safety_margin_mb=512),
        api=APIConfig(),
        models=ModelsConfig(
            llm=[
                LLMModelConfig(
                    name="test-model",
                    model_path="/fake/path/model.gguf",
                    vram_mb=4096,
                )
            ]
        ),
    )


@pytest.fixture
def mock_redis_client() -> MagicMock:
    """Create mock Redis client."""
    client = MagicMock()
    client.ping.return_value = True
    return client


@pytest.fixture
def test_client(mock_config: VramCramConfig, mock_redis_client: MagicMock) -> TestClient:
    """Create test client with mocked dependencies."""
    with patch("vramcram.api.gateway.RedisClientFactory") as mock_factory_class:
        mock_factory = MagicMock(spec=RedisClientFactory)
        mock_factory.create_client.return_value = mock_redis_client
        mock_factory_class.return_value = mock_factory

        with patch("vramcram.api.gateway.EventBus") as mock_bus_class:
            mock_bus = MagicMock(spec=EventBus)
            mock_bus_class.return_value = mock_bus

            app = create_app(mock_config, mock_factory, mock_bus)
            return TestClient(app)


class TestAPIGateway:
    """Test suite for API Gateway."""

    def test_submit_job_success(
        self, test_client: TestClient, mock_redis_client: MagicMock
    ) -> None:
        """Test successful job submission."""
        response = test_client.post(
            "/jobs",
            json={
                "model": "test-model",
                "prompt": "test prompt",
                "params": {"max_tokens": 100},
            },
        )

        assert response.status_code == 201
        data = response.json()

        assert "job_id" in data
        assert data["model"] == "test-model"
        assert data["status"] == "queued"
        assert "created_at" in data

        # Verify Redis calls
        assert mock_redis_client.hset.called
        assert mock_redis_client.xadd.called

    def test_submit_job_invalid_model(self, test_client: TestClient) -> None:
        """Test job submission with invalid model."""
        response = test_client.post(
            "/jobs",
            json={
                "model": "nonexistent-model",
                "prompt": "test prompt",
                "params": {},
            },
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_job_status_success(
        self, test_client: TestClient, mock_redis_client: MagicMock
    ) -> None:
        """Test getting job status."""
        # Mock Redis response
        mock_redis_client.hgetall.return_value = {
            "job_id": "test-123",
            "model": "test-model",
            "prompt": "test",
            "params": "{}",
            "status": "completed",
            "created_at": "2024-01-01T00:00:00+00:00",
            "dispatched_at": None,
            "started_at": None,
            "completed_at": "2024-01-01T00:00:10+00:00",
            "result": None,
            "error": None,
            "duration_ms": "1000",
        }

        response = test_client.get("/jobs/test-123")

        assert response.status_code == 200
        data = response.json()

        assert data["job_id"] == "test-123"
        assert data["model"] == "test-model"
        assert data["status"] == "completed"
        assert data["duration_ms"] == 1000

    def test_get_job_status_not_found(
        self, test_client: TestClient, mock_redis_client: MagicMock
    ) -> None:
        """Test getting status for nonexistent job."""
        mock_redis_client.hgetall.return_value = {}

        response = test_client.get("/jobs/nonexistent")

        assert response.status_code == 404

    def test_get_job_result_success(
        self, test_client: TestClient, mock_redis_client: MagicMock
    ) -> None:
        """Test getting job result."""
        # Mock job status
        mock_redis_client.hgetall.return_value = {
            "job_id": "test-123",
            "model": "test-model",
            "prompt": "test",
            "params": "{}",
            "status": "completed",
            "created_at": "2024-01-01T00:00:00+00:00",
            "dispatched_at": None,
            "started_at": None,
            "completed_at": "2024-01-01T00:00:10+00:00",
            "result": '{"text": "result text"}',
            "error": None,
            "duration_ms": "1000",
        }

        # Mock result
        mock_redis_client.get.return_value = '{"text": "result text"}'

        response = test_client.get("/jobs/test-123/result")

        assert response.status_code == 200
        data = response.json()

        assert data["job_id"] == "test-123"
        assert data["status"] == "completed"
        assert data["result"] == '{"text": "result text"}'
        assert data["error"] is None

    def test_get_job_result_not_completed(
        self, test_client: TestClient, mock_redis_client: MagicMock
    ) -> None:
        """Test getting result for incomplete job."""
        mock_redis_client.hgetall.return_value = {
            "job_id": "test-123",
            "model": "test-model",
            "prompt": "test",
            "params": "{}",
            "status": "processing",
            "created_at": "2024-01-01T00:00:00+00:00",
            "dispatched_at": None,
            "started_at": "2024-01-01T00:00:05+00:00",
            "completed_at": None,
            "result": None,
            "error": None,
            "duration_ms": None,
        }

        response = test_client.get("/jobs/test-123/result")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "processing"
        assert data["result"] is None

    def test_list_models(self, test_client: TestClient) -> None:
        """Test listing models."""
        response = test_client.get("/models")

        assert response.status_code == 200
        data = response.json()

        assert len(data) == 1
        assert data[0]["name"] == "test-model"
        assert data[0]["type"] == "llm"
        assert data[0]["vram_mb"] == 4096

    def test_health_check_healthy(
        self, test_client: TestClient, mock_redis_client: MagicMock
    ) -> None:
        """Test health check when system is healthy."""
        mock_redis_client.ping.return_value = True
        mock_redis_client.xinfo_stream.return_value = {"length": 5}

        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["redis_connected"] is True
        assert data["models_available"] == 1
        assert data["jobs_queued"] == 5

    def test_health_check_unhealthy(
        self, test_client: TestClient, mock_redis_client: MagicMock
    ) -> None:
        """Test health check when system is unhealthy."""
        mock_redis_client.ping.side_effect = Exception("Connection failed")

        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "unhealthy"
        assert data["redis_connected"] is False
