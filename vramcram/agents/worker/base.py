"""Base worker agent for inference execution."""

import asyncio
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import structlog

from vramcram.config.models import VramCramConfig
from vramcram.events.bus import EventBus
from vramcram.events.schema import AgentEvent
from vramcram.events.types import EventType
from vramcram.queue.job import Job, JobStatus
from vramcram.redis.client import RedisClientFactory

logger = structlog.get_logger()


class BaseWorker(ABC):
    """Base class for worker agents.

    Workers are responsible for:
    - Loading inference models (abstract method)
    - Self-assigning jobs from Redis Streams using consumer groups
    - Executing inference (abstract method)
    - Storing results in Redis with TTL
    - Publishing heartbeat events
    - Acknowledging completed jobs

    Lifecycle (from spec lines 400-536):
    1. Load model via abstract load_inference_model()
    2. Publish WORKER_READY
    3. Start heartbeat task (every 10s)
    4. Self-assignment loop: XREADGROUP from jobs:stream
    5. Execute inference, store result, update job status, ACK

    Job execution:
    - Pull job from Redis Stream (consumer group: workers-{model_name})
    - Execute via abstract execute_inference(prompt, params)
    - Store result: result:{job_id} with 24h TTL
    - Update job status: job:{job_id} → "completed" with duration
    - Publish WORKER_JOB_COMPLETED
    - Acknowledge with XACK
    """

    def __init__(
        self,
        worker_id: str,
        model_name: str,
        event_bus: EventBus,
        redis_factory: RedisClientFactory,
        config: VramCramConfig,
    ) -> None:
        """Initialize worker.

        Args:
            worker_id: Unique worker identifier.
            model_name: Name of model this worker handles.
            event_bus: Event bus for communication.
            redis_factory: Redis client factory.
            config: System configuration.
        """
        self.worker_id = worker_id
        self.model_name = model_name
        self.event_bus = event_bus
        self.redis_factory = redis_factory
        self.redis_client = redis_factory.create_client()
        self.config = config
        self.logger = logger.bind(worker_id=worker_id, model=model_name)

        self._shutdown_requested = False
        self._model_loaded = False

        # Consumer group and stream names
        self.stream_name = "jobs:stream"
        self.consumer_group = f"workers-{model_name}"
        self.consumer_name = f"worker-{os.getpid()}"

    async def run(self) -> None:
        """Run worker main loop."""
        self.logger.info("worker_starting", pid=os.getpid())

        try:
            # Load model
            self.logger.info("loading_inference_model")
            await self.load_inference_model()
            self._model_loaded = True
            self.logger.info("inference_model_loaded")

            # Publish WORKER_READY
            self._publish_event(
                EventType.WORKER_READY,
                {"model_name": self.model_name, "worker_pid": os.getpid()},
            )

            # Ensure consumer group exists
            self._ensure_consumer_group()

            # Start background tasks
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            job_task = asyncio.create_task(self._job_processing_loop())

            await asyncio.gather(heartbeat_task, job_task)

        except Exception:
            self.logger.exception("worker_error")
            raise
        finally:
            self.logger.info("worker_stopped")

    @abstractmethod
    async def load_inference_model(self) -> None:
        """Load the inference model.

        This method must be implemented by subclasses to load the specific
        model type (LLM, diffusion, etc.) into memory.

        Raises:
            Exception: If model loading fails.
        """
        pass

    @abstractmethod
    async def execute_inference(
        self, prompt: str, params: dict[str, Any]
    ) -> str:
        """Execute inference on the loaded model.

        Args:
            prompt: Input prompt/text.
            params: Model-specific parameters.

        Returns:
            String containing inference result (text or base64 data URI).
            For LLMs: Direct text string
            For diffusion: Base64 data URI or filesystem path (based on config)

        Raises:
            Exception: If inference fails.
        """
        pass

    def _ensure_consumer_group(self) -> None:
        """Ensure Redis consumer group exists for this model."""
        try:
            # Try to create consumer group
            self.redis_client.xgroup_create(
                name=self.stream_name,
                groupname=self.consumer_group,
                id="0",
                mkstream=True,
            )
            self.logger.info("consumer_group_created", group=self.consumer_group)
        except Exception as e:
            # Group already exists - this is fine
            self.logger.debug("consumer_group_exists", group=self.consumer_group, error=str(e))

    async def _job_processing_loop(self) -> None:
        """Main loop for self-assigning and processing jobs."""
        while not self._shutdown_requested:
            try:
                # Read from stream with consumer group
                # Block for 1 second, read 1 job at a time
                jobs = self.redis_client.xreadgroup(
                    groupname=self.consumer_group,
                    consumername=self.consumer_name,
                    streams={self.stream_name: ">"},
                    count=1,
                    block=1000,
                )

                if not jobs:
                    # No jobs available
                    await asyncio.sleep(0.1)
                    continue

                # Process job
                for stream_name, messages in jobs:
                    for message_id, message_data in messages:
                        await self._process_job(message_id, message_data)

            except Exception:
                self.logger.exception("job_processing_loop_error")
                await asyncio.sleep(1)

    async def _process_job(self, message_id: str, message_data: dict[str, Any]) -> None:
        """Process a single job.

        Args:
            message_id: Redis stream message ID.
            message_data: Job data from stream.
        """
        job_id = message_data.get("job_id")

        if not job_id:
            self.logger.warning("job_missing_id", message_data=message_data)
            # ACK anyway to remove from stream
            self.redis_client.xack(self.stream_name, self.consumer_group, message_id)
            return

        self.logger.info("processing_job", job_id=job_id, message_id=message_id)

        try:
            # Get full job details from Redis
            job_key = f"job:{job_id}"
            job_data = self.redis_client.hgetall(job_key)

            if not job_data:
                self.logger.error("job_not_found", job_id=job_id)
                self.redis_client.xack(self.stream_name, self.consumer_group, message_id)
                return

            # Parse job
            job = Job.from_dict(job_data)

            # Check if job is for this model
            if job.model != self.model_name:
                self.logger.warning(
                    "job_wrong_model",
                    job_id=job_id,
                    expected=self.model_name,
                    actual=job.model,
                )
                self.redis_client.xack(self.stream_name, self.consumer_group, message_id)
                return

            # Update job status to PROCESSING
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.now(timezone.utc)
            self.redis_client.hset(job_key, mapping=job.to_dict())

            # Publish event
            self._publish_event(
                EventType.WORKER_JOB_STARTED,
                {"job_id": job_id, "model": self.model_name},
            )

            # Execute inference (now returns string directly)
            start_time = time.time()
            result = await self.execute_inference(job.prompt, job.params)
            duration_ms = int((time.time() - start_time) * 1000)

            # Store result string directly in Redis with TTL
            result_key = f"result:{job_id}"
            result_ttl = self.config.jobs.result_ttl_seconds
            self.redis_client.setex(result_key, result_ttl, result)

            # Update job status to COMPLETED
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            job.duration_ms = duration_ms
            job.result = result
            self.redis_client.hset(job_key, mapping=job.to_dict())

            # Publish event
            self._publish_event(
                EventType.WORKER_JOB_COMPLETED,
                {
                    "job_id": job_id,
                    "model": self.model_name,
                    "duration_ms": duration_ms,
                },
            )

            self.logger.info(
                "job_completed",
                job_id=job_id,
                duration_ms=duration_ms,
            )

        except Exception as e:
            self.logger.exception("job_execution_failed", job_id=job_id)

            # Update job status to FAILED
            try:
                job_key = f"job:{job_id}"
                job_data = self.redis_client.hgetall(job_key)
                if job_data:
                    job = Job.from_dict(job_data)
                    job.status = JobStatus.FAILED
                    job.completed_at = datetime.now(timezone.utc)
                    job.error = str(e)
                    self.redis_client.hset(job_key, mapping=job.to_dict())

                # Publish event
                self._publish_event(
                    EventType.WORKER_JOB_FAILED,
                    {
                        "job_id": job_id,
                        "model": self.model_name,
                        "error": str(e),
                    },
                )
            except Exception:
                self.logger.exception("failed_to_update_job_status", job_id=job_id)

        finally:
            # Always ACK the message to remove from stream
            try:
                self.redis_client.xack(self.stream_name, self.consumer_group, message_id)
                self.logger.debug("job_acknowledged", job_id=job_id, message_id=message_id)
            except Exception:
                self.logger.exception("failed_to_ack_job", job_id=job_id)

    async def _heartbeat_loop(self) -> None:
        """Publish heartbeat periodically."""
        interval = self.config.agents.heartbeat_interval_seconds

        while not self._shutdown_requested:
            self._publish_event(
                EventType.WORKER_HEARTBEAT,
                {
                    "model_name": self.model_name,
                    "worker_pid": os.getpid(),
                    "model_loaded": self._model_loaded,
                },
            )

            self.logger.debug("heartbeat_published")

            await asyncio.sleep(interval)

    def _publish_event(self, event_type: EventType, payload: dict[str, Any]) -> None:
        """Publish an event.

        Args:
            event_type: Event type.
            payload: Event payload.
        """
        event = AgentEvent(
            event_type=event_type,
            source_agent_id=self.worker_id,
            payload=payload,
        )
        self.event_bus.publish(event)

    def stop(self) -> None:
        """Request shutdown."""
        self.logger.info("worker_shutdown_requested")
        self._shutdown_requested = True

        # Publish shutdown event
        self._publish_event(
            EventType.WORKER_SHUTDOWN,
            {"model_name": self.model_name, "worker_pid": os.getpid()},
        )
