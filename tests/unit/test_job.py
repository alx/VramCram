"""Unit tests for Job dataclass."""

import json
from datetime import datetime, timezone

import pytest

from vramcram.queue.job import Job, JobStatus


class TestJob:
    """Test suite for Job."""

    def test_job_creation(self) -> None:
        """Test creating a job."""
        job = Job(
            job_id="test-123",
            model="test-model",
            prompt="test prompt",
            params={"key": "value"},
        )

        assert job.job_id == "test-123"
        assert job.model == "test-model"
        assert job.prompt == "test prompt"
        assert job.params == {"key": "value"}
        assert job.status == JobStatus.QUEUED
        assert isinstance(job.created_at, datetime)

    def test_job_to_dict(self) -> None:
        """Test serializing job to dict."""
        now = datetime.now(timezone.utc)
        job = Job(
            job_id="test-123",
            model="test-model",
            prompt="test prompt",
            params={"max_tokens": 100},
            status=JobStatus.COMPLETED,
            created_at=now,
        )

        data = job.to_dict()

        assert data["job_id"] == "test-123"
        assert data["model"] == "test-model"
        assert data["prompt"] == "test prompt"
        # params should be JSON serialized for Redis storage
        assert data["params"] == '{"max_tokens": 100}'
        assert data["status"] == "completed"
        assert data["created_at"] == now.isoformat()

    def test_job_to_json(self) -> None:
        """Test serializing job to JSON."""
        job = Job(
            job_id="test-123",
            model="test-model",
            prompt="test prompt",
        )

        json_str = job.to_json()
        data = json.loads(json_str)

        assert data["job_id"] == "test-123"
        assert data["model"] == "test-model"
        assert data["status"] == "queued"

    def test_job_from_dict(self) -> None:
        """Test deserializing job from dict."""
        now = datetime.now(timezone.utc)
        data = {
            "job_id": "test-123",
            "model": "test-model",
            "prompt": "test prompt",
            "params": {"max_tokens": 100},
            "status": "processing",
            "created_at": now.isoformat(),
            "dispatched_at": None,
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
            "duration_ms": None,
        }

        job = Job.from_dict(data)

        assert job.job_id == "test-123"
        assert job.model == "test-model"
        assert job.prompt == "test prompt"
        assert job.params == {"max_tokens": 100}
        assert job.status == JobStatus.PROCESSING
        assert job.created_at == now

    def test_job_from_json(self) -> None:
        """Test deserializing job from JSON."""
        json_data = {
            "job_id": "test-123",
            "model": "test-model",
            "prompt": "test prompt",
            "params": {},
            "status": "queued",
            "created_at": "2024-01-01T00:00:00+00:00",
            "dispatched_at": None,
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
            "duration_ms": None,
        }

        json_str = json.dumps(json_data)
        job = Job.from_json(json_str)

        assert job.job_id == "test-123"
        assert job.model == "test-model"
        assert job.status == JobStatus.QUEUED

    def test_job_from_dict_invalid(self) -> None:
        """Test deserializing invalid job data raises error."""
        with pytest.raises(ValueError, match="Invalid job data"):
            Job.from_dict({"invalid": "data"})

    def test_job_from_json_invalid(self) -> None:
        """Test deserializing invalid JSON raises error."""
        with pytest.raises(ValueError, match="Invalid job JSON"):
            Job.from_json("not json")

    def test_job_status_enum(self) -> None:
        """Test JobStatus enum values."""
        assert JobStatus.QUEUED.value == "queued"
        assert JobStatus.DISPATCHED.value == "dispatched"
        assert JobStatus.PROCESSING.value == "processing"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
