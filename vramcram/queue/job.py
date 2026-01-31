"""Job data structures and serialization."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    """Job processing status."""

    QUEUED = "queued"
    DISPATCHED = "dispatched"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    """Represents an inference job.

    Jobs flow through the system as follows:
    1. Created via API with status QUEUED
    2. Dispatched by Coordinator to a worker with status DISPATCHED
    3. Worker picks up and sets status to PROCESSING
    4. Worker completes and sets status to COMPLETED or FAILED

    Attributes:
        job_id: Unique job identifier.
        model: Target model name.
        prompt: Inference prompt/input.
        params: Model-specific parameters.
        status: Current job status.
        created_at: Job creation timestamp.
        dispatched_at: When job was dispatched to a worker.
        started_at: When worker started processing.
        completed_at: When processing finished.
        result: Result data (prompt response or image path).
        error: Error message if job failed.
        duration_ms: Processing duration in milliseconds.
    """

    job_id: str
    model: str
    prompt: str
    params: dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    dispatched_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: str | None = None
    error: str | None = None
    duration_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize job to dictionary for Redis storage.

        Returns:
            Dictionary representation suitable for Redis hset.
            All None values are converted to empty strings for Redis compatibility.
        """
        data = asdict(self)
        # Convert enums and datetime objects to strings
        data["status"] = self.status.value
        data["created_at"] = self.created_at.isoformat()
        data["dispatched_at"] = (
            self.dispatched_at.isoformat() if self.dispatched_at else ""
        )
        data["started_at"] = self.started_at.isoformat() if self.started_at else ""
        data["completed_at"] = (
            self.completed_at.isoformat() if self.completed_at else ""
        )
        data["result"] = self.result if self.result else ""
        data["error"] = self.error if self.error else ""
        data["duration_ms"] = str(self.duration_ms) if self.duration_ms else ""
        # Serialize params dict to JSON for Redis storage
        data["params"] = json.dumps(self.params)
        return data

    def to_json(self) -> str:
        """Serialize job to JSON string.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Job":
        """Deserialize job from dictionary.

        Args:
            data: Dictionary representation from Redis.

        Returns:
            Job instance.

        Raises:
            ValueError: If data is invalid or missing required fields.
        """
        try:
            # Convert string status to enum
            if isinstance(data.get("status"), str):
                data["status"] = JobStatus(data["status"])

            # Deserialize params from JSON if it's a string
            if isinstance(data.get("params"), str):
                data["params"] = json.loads(data["params"])

            # Convert empty strings to None for optional fields
            data["dispatched_at"] = data.get("dispatched_at") or None
            data["started_at"] = data.get("started_at") or None
            data["completed_at"] = data.get("completed_at") or None
            data["result"] = data.get("result") or None
            data["error"] = data.get("error") or None
            data["duration_ms"] = data.get("duration_ms") or None

            # Convert ISO format strings to datetime
            if isinstance(data.get("created_at"), str):
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            if data.get("dispatched_at"):
                data["dispatched_at"] = datetime.fromisoformat(data["dispatched_at"])
            if data.get("started_at"):
                data["started_at"] = datetime.fromisoformat(data["started_at"])
            if data.get("completed_at"):
                data["completed_at"] = datetime.fromisoformat(data["completed_at"])

            # Convert duration_ms to int if present
            if data.get("duration_ms"):
                data["duration_ms"] = int(data["duration_ms"])

            return cls(**data)
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid job data: {e}") from e

    @classmethod
    def from_json(cls, json_str: str) -> "Job":
        """Deserialize job from JSON string.

        Args:
            json_str: JSON string representation.

        Returns:
            Job instance.

        Raises:
            ValueError: If JSON is invalid or missing required fields.
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid job JSON: {e}") from e
