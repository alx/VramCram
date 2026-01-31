"""FastAPI gateway for VramCram REST API."""

import uuid
from datetime import datetime, timezone

import structlog
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from vramcram.api.models import (
    HealthResponse,
    JobResultResponse,
    JobStatusResponse,
    JobSubmitRequest,
    JobSubmitResponse,
    ModelInfo,
)
from vramcram.config.models import VramCramConfig
from vramcram.events.bus import EventBus
from vramcram.events.schema import AgentEvent
from vramcram.events.types import EventType
from vramcram.queue.job import Job, JobStatus
from vramcram.redis.client import RedisClientFactory

logger = structlog.get_logger()


class APIGateway:
    """FastAPI gateway for job submission and status queries.

    Endpoints (from spec lines 799-889):
    - POST /jobs: Generate job_id, write to Redis Stream, store metadata, publish JOB_QUEUED
    - GET /jobs/{job_id}: Return status from Redis hash
    - GET /jobs/{job_id}/result: Return result if status is "completed"
    - GET /models: List models from config
    - GET /health: Check Redis connection, return system state
    """

    def __init__(
        self,
        config: VramCramConfig,
        redis_factory: RedisClientFactory,
        event_bus: EventBus,
    ) -> None:
        """Initialize API gateway.

        Args:
            config: System configuration.
            redis_factory: Redis client factory.
            event_bus: Event bus for publishing events.
        """
        self.config = config
        self.redis_factory = redis_factory
        self.redis_client = redis_factory.create_client()
        self.event_bus = event_bus
        self.logger = logger.bind(component="api_gateway")

        # Create FastAPI app
        self.app = FastAPI(
            title="VramCram API",
            description="GPU orchestration system for LLM and Diffusion models",
            version="0.1.0",
        )

        # Register routes
        self._register_routes()

    def _register_routes(self) -> None:
        """Register API routes."""

        @self.app.post("/jobs", response_model=JobSubmitResponse, status_code=status.HTTP_201_CREATED)
        async def submit_job(request: JobSubmitRequest) -> JobSubmitResponse:
            """Submit a new job for processing.

            Args:
                request: Job submission request.

            Returns:
                Job submission response with job_id.

            Raises:
                HTTPException: If model not found or job submission fails.
            """
            try:
                # Validate model exists
                if not self._model_exists(request.model):
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Model not found: {request.model}",
                    )

                # Generate job ID
                job_id = str(uuid.uuid4())

                # Create job
                job = Job(
                    job_id=job_id,
                    model=request.model,
                    prompt=request.prompt,
                    params=request.params,
                    status=JobStatus.QUEUED,
                    created_at=datetime.now(timezone.utc),
                )

                # Store job metadata in Redis hash
                job_key = f"job:{job_id}"
                self.redis_client.hset(job_key, mapping=job.to_dict())

                # Add job to Redis Stream
                stream_name = "jobs:stream"
                self.redis_client.xadd(
                    stream_name,
                    {
                        "job_id": job_id,
                        "model": request.model,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )

                # Publish JOB_QUEUED event
                event = AgentEvent(
                    event_type=EventType.JOB_QUEUED,
                    source_agent_id="api_gateway",
                    payload=job.to_dict(),
                )
                self.logger.info("publishing_job_queued_event", job_id=job_id, channel=f"events:{EventType.JOB_QUEUED.value}")
                self.event_bus.publish(event)
                self.logger.info("job_queued_event_published", job_id=job_id)

                self.logger.info("job_submitted", job_id=job_id, model=request.model)

                return JobSubmitResponse(
                    job_id=job_id,
                    model=request.model,
                    status=job.status.value,
                    created_at=job.created_at,
                )

            except HTTPException:
                raise
            except Exception as e:
                self.logger.exception("job_submission_failed")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Job submission failed: {str(e)}",
                ) from e

        @self.app.get("/jobs/{job_id}", response_model=JobStatusResponse)
        async def get_job_status(job_id: str) -> JobStatusResponse:
            """Get job status.

            Args:
                job_id: Job identifier.

            Returns:
                Job status information.

            Raises:
                HTTPException: If job not found.
            """
            try:
                job_key = f"job:{job_id}"
                job_data = self.redis_client.hgetall(job_key)

                if not job_data:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Job not found: {job_id}",
                    )

                job = Job.from_dict(job_data)

                return JobStatusResponse(
                    job_id=job.job_id,
                    model=job.model,
                    status=job.status.value,
                    created_at=job.created_at,
                    dispatched_at=job.dispatched_at,
                    started_at=job.started_at,
                    completed_at=job.completed_at,
                    duration_ms=job.duration_ms,
                    error=job.error,
                )

            except HTTPException:
                raise
            except Exception as e:
                self.logger.exception("get_job_status_failed", job_id=job_id)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get job status: {str(e)}",
                ) from e

        @self.app.get("/jobs/{job_id}/result", response_model=JobResultResponse)
        async def get_job_result(job_id: str) -> JobResultResponse:
            """Get job result.

            Args:
                job_id: Job identifier.

            Returns:
                Job result if completed.

            Raises:
                HTTPException: If job not found or not completed.
            """
            try:
                job_key = f"job:{job_id}"
                job_data = self.redis_client.hgetall(job_key)

                if not job_data:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Job not found: {job_id}",
                    )

                job = Job.from_dict(job_data)

                if job.status != JobStatus.COMPLETED:
                    return JobResultResponse(
                        job_id=job.job_id,
                        status=job.status.value,
                        result=None,
                        error=job.error if job.status == JobStatus.FAILED else None,
                    )

                # Get result from Redis
                result_key = f"result:{job_id}"
                result = self.redis_client.get(result_key)

                return JobResultResponse(
                    job_id=job.job_id,
                    status=job.status.value,
                    result=result,
                    error=None,
                )

            except HTTPException:
                raise
            except Exception as e:
                self.logger.exception("get_job_result_failed", job_id=job_id)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get job result: {str(e)}",
                ) from e

        @self.app.get("/models", response_model=list[ModelInfo])
        async def list_models() -> list[ModelInfo]:
            """List available models.

            Returns:
                List of model information.
            """
            try:
                models = []

                # Add LLM models
                for model in self.config.models.llm:
                    models.append(
                        ModelInfo(
                            name=model.name,
                            type="llm",
                            vram_mb=model.vram_mb,
                            loaded=False,  # TODO: Query coordinator's registry
                        )
                    )

                # Add diffusion models
                for model in self.config.models.diffusion:
                    models.append(
                        ModelInfo(
                            name=model.name,
                            type="diffusion",
                            vram_mb=model.vram_mb,
                            loaded=False,  # TODO: Query coordinator's registry
                        )
                    )

                return models

            except Exception as e:
                self.logger.exception("list_models_failed")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to list models: {str(e)}",
                ) from e

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check() -> HealthResponse:
            """Health check endpoint.

            Returns:
                System health status.
            """
            try:
                # Check Redis connection
                redis_connected = self.redis_client.ping()

                # Count models
                models_available = len(self.config.models.llm) + len(
                    self.config.models.diffusion
                )

                # Get queue size (approximate)
                try:
                    stream_info = self.redis_client.xinfo_stream("jobs:stream")
                    jobs_queued = stream_info.get("length", 0)
                except Exception:
                    jobs_queued = 0

                status_str = "healthy" if redis_connected else "unhealthy"

                return HealthResponse(
                    status=status_str,
                    redis_connected=redis_connected,
                    models_available=models_available,
                    jobs_queued=jobs_queued,
                )

            except Exception as e:
                self.logger.exception("health_check_failed")
                return HealthResponse(
                    status="unhealthy",
                    redis_connected=False,
                    models_available=0,
                    jobs_queued=0,
                )

    def _model_exists(self, model_name: str) -> bool:
        """Check if a model exists in configuration.

        Args:
            model_name: Model name to check.

        Returns:
            True if model exists, False otherwise.
        """
        for model in self.config.models.llm:
            if model.name == model_name:
                return True
        for model in self.config.models.diffusion:
            if model.name == model_name:
                return True
        return False


def create_app(
    config: VramCramConfig, redis_factory: RedisClientFactory, event_bus: EventBus
) -> FastAPI:
    """Create FastAPI application.

    Args:
        config: System configuration.
        redis_factory: Redis client factory.
        event_bus: Event bus.

    Returns:
        FastAPI application instance.
    """
    gateway = APIGateway(config, redis_factory, event_bus)
    return gateway.app
