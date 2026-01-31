"""Coordinator agent for system-wide orchestration."""

import asyncio
from collections import deque
from datetime import datetime, timezone

import structlog

from vramcram.config.models import VramCramConfig
from vramcram.events.bus import EventBus
from vramcram.events.schema import AgentEvent
from vramcram.events.types import EventType
from vramcram.gpu.vram_tracker import VRAMTracker
from vramcram.queue.job import Job
from vramcram.queue.registry import ModelRegistry, ModelState

logger = structlog.get_logger()


class CoordinatorAgent:
    """Coordinator agent for FIFO job queue and VRAM orchestration.

    The Coordinator is responsible for:
    - Maintaining a FIFO queue of pending jobs
    - Monitoring VRAM availability
    - Making LRU eviction decisions when VRAM is insufficient
    - Dispatching jobs to workers
    - Tracking agent heartbeats for health monitoring

    Algorithm (from spec lines 84-96):
    1. Poll VRAM every 5 seconds
    2. Process job queue:
       - Peek next job
       - Check if model is loaded
       - If loaded: dispatch job, update last_used
       - If not loaded and sufficient VRAM: request load
       - If not loaded and insufficient VRAM: select LRU victim, request eviction
    3. Check heartbeats (timeout 30s)
    """

    def __init__(
        self,
        agent_id: str,
        event_bus: EventBus,
        config: VramCramConfig,
        vram_tracker: VRAMTracker,
    ) -> None:
        """Initialize coordinator agent.

        Args:
            agent_id: Unique agent identifier.
            event_bus: Event bus for communication.
            config: System configuration.
            vram_tracker: VRAM monitoring instance.
        """
        self.agent_id = agent_id
        self.event_bus = event_bus
        self.config = config
        self.vram_tracker = vram_tracker
        self.logger = logger.bind(agent_id=agent_id)

        # Core state
        self.job_queue: deque[Job] = deque()
        self.model_registry = ModelRegistry()
        self.heartbeats: dict[str, datetime] = {}
        self._shutdown_requested = False

        # Model configs lookup
        self.model_configs: dict[str, int] = {}
        for model in config.models.llm:
            self.model_configs[model.name] = model.vram_mb
        for model in config.models.diffusion:
            self.model_configs[model.name] = model.vram_mb

    async def run(self) -> None:
        """Run coordinator main loop."""
        self.logger.info("coordinator_starting")

        # Subscribe to relevant events
        self.event_bus.subscribe(
            EventType.JOB_QUEUED,
            EventType.MODEL_LOADED,
            EventType.MODEL_UNLOADED,
            EventType.WORKER_HEARTBEAT,
            EventType.AGENT_HEARTBEAT,
        )

        # Start background tasks
        vram_task = asyncio.create_task(self._vram_monitor_loop())
        heartbeat_task = asyncio.create_task(self._heartbeat_check_loop())
        queue_task = asyncio.create_task(self._queue_processor_loop())
        event_task = asyncio.create_task(self._event_listener_loop())

        try:
            await asyncio.gather(vram_task, heartbeat_task, queue_task, event_task)
        except asyncio.CancelledError:
            self.logger.info("coordinator_cancelled")
        finally:
            self.logger.info("coordinator_stopped")

    async def _event_listener_loop(self) -> None:
        """Listen for events from event bus."""
        # Run blocking listen() in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()

        while not self._shutdown_requested:
            try:
                # Get next event with timeout to allow checking shutdown flag
                event = await loop.run_in_executor(
                    None,
                    lambda: self.event_bus.wait_for_event(timeout=1.0)
                )

                if event:
                    await self._handle_event(event)
            except Exception:
                self.logger.exception("event_listener_error")
                await asyncio.sleep(0.1)

    async def _handle_event(self, event: AgentEvent) -> None:
        """Handle incoming events.

        Args:
            event: Event to handle.
        """
        if event.event_type == EventType.JOB_QUEUED:
            await self._handle_job_queued(event)
        elif event.event_type == EventType.MODEL_LOADED:
            await self._handle_model_loaded(event)
        elif event.event_type == EventType.MODEL_UNLOADED:
            await self._handle_model_unloaded(event)
        elif event.event_type in (EventType.WORKER_HEARTBEAT, EventType.AGENT_HEARTBEAT):
            await self._handle_heartbeat(event)

    async def _handle_job_queued(self, event: AgentEvent) -> None:
        """Handle job queued event.

        Args:
            event: Job queued event.
        """
        job_data = event.payload
        job = Job.from_dict(job_data)

        self.logger.info(
            "job_queued",
            job_id=job.job_id,
            model=job.model,
            queue_size=len(self.job_queue),
        )

        self.job_queue.append(job)

    async def _handle_model_loaded(self, event: AgentEvent) -> None:
        """Handle model loaded event.

        Args:
            event: Model loaded event.
        """
        model_name = event.payload.get("model_name")
        worker_pid = event.payload.get("worker_pid")

        if not model_name:
            self.logger.warning("model_loaded_missing_name", payload=event.payload)
            return

        vram_mb = self.model_configs.get(model_name, 0)

        self.model_registry.add_model(
            model_name=model_name,
            vram_mb=vram_mb,
            worker_pid=worker_pid,
            state=ModelState.LOADED,
        )

        self.logger.info(
            "model_loaded_registered",
            model=model_name,
            vram_mb=vram_mb,
            worker_pid=worker_pid,
        )

    async def _handle_model_unloaded(self, event: AgentEvent) -> None:
        """Handle model unloaded event.

        Args:
            event: Model unloaded event.
        """
        model_name = event.payload.get("model_name")

        if not model_name:
            self.logger.warning("model_unloaded_missing_name", payload=event.payload)
            return

        self.model_registry.remove_model(model_name)

        self.logger.info("model_unloaded_deregistered", model=model_name)

    async def _handle_heartbeat(self, event: AgentEvent) -> None:
        """Handle heartbeat event.

        Args:
            event: Heartbeat event.
        """
        source_id = event.source_agent_id
        self.heartbeats[source_id] = datetime.now(timezone.utc)

        self.logger.debug("heartbeat_received", source=source_id)

    async def _vram_monitor_loop(self) -> None:
        """Monitor VRAM periodically."""
        interval = self.config.vram.monitoring_interval_seconds

        while not self._shutdown_requested:
            try:
                vram_state = self.vram_tracker.get_vram_state()

                self.logger.debug(
                    "vram_monitored",
                    free_mb=vram_state.free_mb,
                    used_mb=vram_state.used_mb,
                    total_mb=vram_state.total_mb,
                )

                # Check for VRAM conditions
                safety_margin = self.config.vram.safety_margin_mb
                available = vram_state.free_mb - safety_margin

                if available < 0:
                    self._publish_event(EventType.VRAM_CRITICAL, {"available_mb": available})
                elif available < 512:  # Arbitrary threshold
                    self._publish_event(EventType.VRAM_LOW, {"available_mb": available})
                else:
                    self._publish_event(EventType.VRAM_AVAILABLE, {"available_mb": available})

            except Exception:
                self.logger.exception("vram_monitor_error")

            await asyncio.sleep(interval)

    async def _heartbeat_check_loop(self) -> None:
        """Check agent heartbeats for timeouts."""
        timeout_seconds = self.config.agents.heartbeat_timeout_seconds

        while not self._shutdown_requested:
            try:
                now = datetime.now(timezone.utc)

                for agent_id, last_heartbeat in list(self.heartbeats.items()):
                    elapsed = (now - last_heartbeat).total_seconds()

                    if elapsed > timeout_seconds:
                        self.logger.warning(
                            "agent_heartbeat_timeout",
                            agent_id=agent_id,
                            elapsed_seconds=elapsed,
                        )

                        # Remove from heartbeats
                        del self.heartbeats[agent_id]

                        # TODO: Handle timeout (e.g., mark model as failed, restart worker)

            except Exception:
                self.logger.exception("heartbeat_check_error")

            await asyncio.sleep(timeout_seconds // 2)

    async def _queue_processor_loop(self) -> None:
        """Process job queue and make dispatch/eviction decisions."""
        while not self._shutdown_requested:
            try:
                if not self.job_queue:
                    await asyncio.sleep(0.5)
                    continue

                # Peek next job (don't remove yet)
                job = self.job_queue[0]

                # Check if model is loaded
                if self.model_registry.is_loaded(job.model):
                    # Model is loaded - dispatch job
                    await self._dispatch_job(job)
                    self.job_queue.popleft()
                elif self.model_registry.is_loading(job.model):
                    # Model is loading - wait for it to finish
                    pass  # Just wait, don't request again
                else:
                    # Model not loaded - check VRAM and request load
                    await self._handle_model_load_request(job)

            except Exception:
                self.logger.exception("queue_processor_error")

            await asyncio.sleep(0.1)

    async def _dispatch_job(self, job: Job) -> None:
        """Dispatch job to worker.

        Args:
            job: Job to dispatch.
        """
        # Update last_used timestamp for LRU tracking
        self.model_registry.update_last_used(job.model)

        # Update job status
        job.dispatched_at = datetime.now(timezone.utc)

        # Publish dispatch event
        self._publish_event(
            EventType.JOB_DISPATCHED,
            {
                "job_id": job.job_id,
                "model": job.model,
                "prompt": job.prompt,
                "params": job.params,
            },
        )

        self.logger.info("job_dispatched", job_id=job.job_id, model=job.model)

    async def _handle_model_load_request(self, job: Job) -> None:
        """Handle model load request with VRAM checking and eviction.

        Args:
            job: Job requiring model load.
        """
        model_name = job.model
        required_mb = self.model_configs.get(model_name)

        if required_mb is None:
            self.logger.error("unknown_model", model=model_name)
            # Remove job from queue as it can't be processed
            self.job_queue.popleft()
            return

        safety_margin = self.config.vram.safety_margin_mb

        # Check if sufficient VRAM available
        if self.vram_tracker.has_sufficient_vram(required_mb, safety_margin):
            # Sufficient VRAM - request load
            self._publish_event(
                EventType.MODEL_LOAD_REQUEST,
                {"model_name": model_name, "job_id": job.job_id},
            )

            # Mark model as loading in registry
            self.model_registry.add_model(
                model_name=model_name,
                vram_mb=required_mb,
                state=ModelState.LOADING,
            )

            self.logger.info("model_load_requested", model=model_name)

        else:
            # Insufficient VRAM - need eviction
            lru_model = self.model_registry.get_lru_model()

            if lru_model:
                self.logger.info(
                    "requesting_eviction",
                    lru_model=lru_model,
                    for_model=model_name,
                )

                self._publish_event(
                    EventType.MODEL_EVICTION_REQUEST,
                    {"model_name": lru_model},
                )

                # Mark model as evicting
                self.model_registry.update_state(lru_model, ModelState.EVICTING)

            else:
                self.logger.warning(
                    "no_lru_model_to_evict",
                    required_mb=required_mb,
                )

    def _publish_event(self, event_type: EventType, payload: dict) -> None:
        """Publish an event.

        Args:
            event_type: Event type.
            payload: Event payload.
        """
        event = AgentEvent(
            event_type=event_type,
            source_agent_id=self.agent_id,
            payload=payload,
        )
        self.event_bus.publish(event)

    def stop(self) -> None:
        """Request shutdown."""
        self.logger.info("coordinator_shutdown_requested")
        self._shutdown_requested = True
