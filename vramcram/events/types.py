"""Event type definitions for VramCram agents."""

from enum import Enum


class EventType(str, Enum):
    """Event types for agent communication."""

    # Job events
    JOB_QUEUED = "job.queued"
    JOB_DISPATCHED = "job.dispatched"
    JOB_PROCESSING = "job.processing"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"

    # Model events
    MODEL_LOAD_REQUEST = "model.load_request"
    MODEL_LOADING = "model.loading"
    MODEL_LOADED = "model.loaded"
    MODEL_LOAD_FAILED = "model.load_failed"
    MODEL_EVICTION_REQUEST = "model.eviction_request"
    MODEL_EVICTING = "model.evicting"
    MODEL_UNLOADED = "model.unloaded"

    # Worker events
    WORKER_READY = "worker.ready"
    WORKER_JOB_STARTED = "worker.job_started"
    WORKER_JOB_COMPLETED = "worker.job_completed"
    WORKER_JOB_FAILED = "worker.job_failed"
    WORKER_HEARTBEAT = "worker.heartbeat"
    WORKER_SHUTDOWN = "worker.shutdown"

    # VRAM events
    VRAM_LOW = "vram.low"
    VRAM_CRITICAL = "vram.critical"
    VRAM_AVAILABLE = "vram.available"

    # Agent health events
    AGENT_STARTED = "agent.started"
    AGENT_HEARTBEAT = "agent.heartbeat"
    AGENT_FAILED = "agent.failed"
    AGENT_SHUTDOWN = "agent.shutdown"
