"""Model Manager agent for per-model lifecycle management."""

import asyncio
import multiprocessing
import os
import signal
from typing import Any

import httpx
import structlog

from vramcram.config.models import VramCramConfig
from vramcram.events.bus import EventBus
from vramcram.events.schema import AgentEvent
from vramcram.events.types import EventType
from vramcram.queue.registry import ModelState

logger = structlog.get_logger()


class ModelManagerAgent:
    """Model Manager for worker lifecycle management.

    Each Model Manager is responsible for a single model and manages:
    - Worker process spawning and termination
    - Model load/unload lifecycle
    - Eviction requests during job processing
    - Heartbeat publishing with model state

    Lifecycle (from spec lines 271-333):
    - Load: Spawn worker process → wait for WORKER_READY (60s timeout) → publish MODEL_LOADED
    - Evict: SIGTERM (30s timeout) → SIGKILL if needed → publish MODEL_UNLOADED

    State machine:
    - unloaded: No worker process
    - loading: Worker process spawned, waiting for WORKER_READY
    - loaded: Worker ready and processing jobs
    - evicting: Worker shutdown in progress
    """

    def __init__(
        self,
        agent_id: str,
        model_name: str,
        event_bus: EventBus,
        config: VramCramConfig,
    ) -> None:
        """Initialize model manager.

        Args:
            agent_id: Unique agent identifier.
            model_name: Name of model this manager handles.
            event_bus: Event bus for communication.
            config: System configuration.
        """
        self.agent_id = agent_id
        self.model_name = model_name
        self.event_bus = event_bus
        self.config = config
        self.logger = logger.bind(agent_id=agent_id, model=model_name)

        # State
        self.state: ModelState = ModelState.UNLOADED
        self.worker_process: multiprocessing.Process | None = None
        self.worker_pid: int | None = None
        self.pending_eviction = False
        self._shutdown_requested = False

        # llama-server process tracking
        self.llama_server_process: asyncio.subprocess.Process | None = None
        self.llama_server_port: int | None = None

    async def run(self) -> None:
        """Run model manager main loop."""
        self.logger.info("model_manager_starting")

        # Subscribe to events
        self.event_bus.subscribe(
            EventType.MODEL_LOAD_REQUEST,
            EventType.MODEL_EVICTION_REQUEST,
            EventType.WORKER_READY,
        )

        # Start background tasks
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        event_task = asyncio.create_task(self._event_listener_loop())

        try:
            await asyncio.gather(heartbeat_task, event_task)
        except asyncio.CancelledError:
            self.logger.info("model_manager_cancelled")
        finally:
            await self._cleanup()
            self.logger.info("model_manager_stopped")

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
        if event.event_type == EventType.MODEL_LOAD_REQUEST:
            await self._handle_load_request(event)
        elif event.event_type == EventType.MODEL_EVICTION_REQUEST:
            await self._handle_eviction_request(event)
        elif event.event_type == EventType.WORKER_READY:
            await self._handle_worker_ready(event)

    async def _handle_load_request(self, event: AgentEvent) -> None:
        """Handle model load request.

        Args:
            event: Load request event.
        """
        requested_model = event.payload.get("model_name")

        # Only handle requests for this model
        if requested_model != self.model_name:
            return

        if self.state != ModelState.UNLOADED:
            self.logger.warning(
                "load_request_invalid_state",
                current_state=self.state.value,
            )
            return

        self.logger.info("handling_load_request")
        await self._load_model()

    async def _handle_eviction_request(self, event: AgentEvent) -> None:
        """Handle model eviction request.

        Args:
            event: Eviction request event.
        """
        requested_model = event.payload.get("model_name")

        # Only handle requests for this model
        if requested_model != self.model_name:
            return

        if self.state != ModelState.LOADED:
            self.logger.warning(
                "eviction_request_invalid_state",
                current_state=self.state.value,
            )
            return

        self.logger.info("handling_eviction_request")
        await self._evict_model()

    async def _handle_worker_ready(self, event: AgentEvent) -> None:
        """Handle worker ready event.

        Args:
            event: Worker ready event.
        """
        worker_model = event.payload.get("model_name")
        worker_pid = event.payload.get("worker_pid")

        # Only handle events for this model
        if worker_model != self.model_name:
            return

        if self.state != ModelState.LOADING:
            self.logger.warning(
                "worker_ready_unexpected_state",
                current_state=self.state.value,
            )
            return

        self.logger.info("worker_ready", worker_pid=worker_pid)

        self.state = ModelState.LOADED
        self.worker_pid = worker_pid

        # Publish MODEL_LOADED
        self._publish_event(
            EventType.MODEL_LOADED,
            {"model_name": self.model_name, "worker_pid": worker_pid},
        )

    def _get_model_index(self) -> int:
        """Get index of this model in LLM models list.

        Returns:
            Model index (0-based).
        """
        for i, model in enumerate(self.config.models.llm):
            if model.name == self.model_name:
                return i
        return 0

    async def _spawn_llama_server(self, model_config: dict[str, Any]) -> int:
        """Spawn llama-server subprocess for this model.

        Args:
            model_config: Model configuration dict.

        Returns:
            Port number llama-server is listening on.

        Raises:
            RuntimeError: If llama-server fails to start.
        """
        # Calculate port (base 8081 + model index)
        model_index = self._get_model_index()
        port = 8081 + model_index

        # Build command
        cmd = [
            str(self.config.inference.llama_server_path),
            "-m", model_config["model_path"],
            "--port", str(port),
            "--host", "127.0.0.1",
            "-c", str(model_config["config"].get("n_ctx", 4096)),
            # Let llama-server auto-fit layers, don't specify -ngl
        ]

        self.logger.info(
            "spawning_llama_server",
            port=port,
            model_path=model_config["model_path"]
        )

        # Spawn subprocess
        self.llama_server_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self.llama_server_port = port

        # Wait for server to be ready (health check)
        server_url = f"http://127.0.0.1:{port}"
        timeout = 30  # seconds

        for i in range(timeout):
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(f"{server_url}/health")
                    if response.status_code == 200:
                        self.logger.info(
                            "llama_server_ready",
                            port=port,
                            elapsed_seconds=i
                        )
                        return port
            except Exception:
                pass  # Server not ready yet

            await asyncio.sleep(1)

        # Timeout - kill process
        self.llama_server_process.terminate()
        raise RuntimeError(
            f"llama-server failed to start within {timeout}s on port {port}"
        )

    async def _load_model(self) -> None:
        """Load model by spawning llama-server and worker process."""
        self.logger.info("loading_model")
        self.state = ModelState.LOADING

        # Publish MODEL_LOADING event
        self._publish_event(EventType.MODEL_LOADING, {"model_name": self.model_name})

        try:
            # Get model config
            model_config = self._get_model_config()
            if model_config is None:
                raise ValueError(f"Model config not found: {self.model_name}")

            # For LLM models, spawn llama-server first
            if model_config["type"] == "llm":
                port = await self._spawn_llama_server(model_config)
                self.logger.info("llama_server_spawned", port=port)

            # Spawn worker process
            # Import here to avoid circular dependency
            from vramcram.agents.worker.llm_worker import llm_worker_main
            from vramcram.agents.worker.diffusion_worker import diffusion_worker_main

            model_type = model_config.get("type")

            if model_type == "llm":
                target = llm_worker_main
            elif model_type == "diffusion":
                target = diffusion_worker_main
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            self.worker_process = multiprocessing.Process(
                target=target,
                args=(self.model_name, self.config),
                name=f"worker-{self.model_name}",
            )

            self.worker_process.start()

            self.logger.info(
                "worker_process_spawned",
                worker_pid=self.worker_process.pid,
            )

            # Wait for WORKER_READY event (handled in _handle_worker_ready)
            # Timeout handled by separate task
            timeout = self.config.agents.worker_ready_timeout_seconds
            asyncio.create_task(self._wait_for_worker_ready(timeout))

        except Exception as e:
            self.logger.exception("model_load_failed")
            self.state = ModelState.UNLOADED

            # Cleanup llama-server if it was spawned
            if self.llama_server_process:
                self.llama_server_process.terminate()
                self.llama_server_process = None
                self.llama_server_port = None

            self._publish_event(
                EventType.MODEL_LOAD_FAILED,
                {"model_name": self.model_name, "error": str(e)},
            )

    async def _wait_for_worker_ready(self, timeout: int) -> None:
        """Wait for worker to become ready with timeout.

        Args:
            timeout: Timeout in seconds.
        """
        await asyncio.sleep(timeout)

        if self.state == ModelState.LOADING:
            self.logger.error("worker_ready_timeout", timeout_seconds=timeout)

            # Kill worker process
            if self.worker_process and self.worker_process.is_alive():
                self.worker_process.terminate()
                await asyncio.sleep(2)
                if self.worker_process.is_alive():
                    self.worker_process.kill()

            self.state = ModelState.UNLOADED
            self.worker_process = None

            self._publish_event(
                EventType.MODEL_LOAD_FAILED,
                {
                    "model_name": self.model_name,
                    "error": "Worker ready timeout",
                },
            )

    async def _evict_model(self) -> None:
        """Evict model by terminating worker process gracefully."""
        self.logger.info("evicting_model")
        self.state = ModelState.EVICTING

        # Publish MODEL_EVICTING event
        self._publish_event(EventType.MODEL_EVICTING, {"model_name": self.model_name})

        if not self.worker_process or not self.worker_process.is_alive():
            self.logger.warning("evict_no_worker_process")
            self.state = ModelState.UNLOADED
            self._publish_event(
                EventType.MODEL_UNLOADED,
                {"model_name": self.model_name},
            )
            return

        try:
            # Send SIGTERM for graceful shutdown
            worker_pid = self.worker_process.pid
            if worker_pid:
                os.kill(worker_pid, signal.SIGTERM)

            self.logger.info("sigterm_sent", worker_pid=worker_pid)

            # Wait for process to exit with timeout
            timeout = self.config.agents.eviction_timeout_seconds

            try:
                # Wait for process to terminate
                for _ in range(timeout * 10):  # Check every 100ms
                    if not self.worker_process.is_alive():
                        break
                    await asyncio.sleep(0.1)

                if self.worker_process.is_alive():
                    # Timeout - send SIGKILL
                    self.logger.warning(
                        "graceful_shutdown_timeout_sigkill",
                        timeout_seconds=timeout,
                    )
                    self.worker_process.kill()
                    await asyncio.sleep(0.5)

            except Exception:
                self.logger.exception("eviction_wait_error")

            self.logger.info("worker_process_terminated")

        except Exception:
            self.logger.exception("eviction_error")

        finally:
            # Kill llama-server if it exists
            if self.llama_server_process:
                try:
                    self.logger.info("terminating_llama_server", port=self.llama_server_port)
                    self.llama_server_process.terminate()

                    # Wait up to 10s for graceful shutdown
                    try:
                        await asyncio.wait_for(
                            self.llama_server_process.wait(),
                            timeout=10.0
                        )
                    except asyncio.TimeoutError:
                        self.logger.warning("llama_server_kill")
                        self.llama_server_process.kill()

                    self.llama_server_process = None
                    self.llama_server_port = None
                except Exception:
                    self.logger.exception("llama_server_cleanup_error")

            self.state = ModelState.UNLOADED
            self.worker_process = None
            self.worker_pid = None

            self._publish_event(
                EventType.MODEL_UNLOADED,
                {"model_name": self.model_name},
            )

    async def _heartbeat_loop(self) -> None:
        """Publish heartbeat periodically."""
        interval = self.config.agents.heartbeat_interval_seconds

        while not self._shutdown_requested:
            self._publish_event(
                EventType.AGENT_HEARTBEAT,
                {
                    "model_name": self.model_name,
                    "state": self.state.value,
                    "worker_pid": self.worker_pid,
                },
            )

            self.logger.debug("heartbeat_published", state=self.state.value)

            await asyncio.sleep(interval)

    def _get_model_config(self) -> dict[str, Any] | None:
        """Get configuration for this model.

        Returns:
            Model config dict or None if not found.
        """
        # Check LLM models
        for model in self.config.models.llm:
            if model.name == self.model_name:
                return {
                    "type": "llm",
                    "model_path": str(model.model_path),
                    "vram_mb": model.vram_mb,
                    "config": model.config,
                }

        # Check diffusion models
        for model in self.config.models.diffusion:
            if model.name == self.model_name:
                return {
                    "type": "diffusion",
                    "model_path": str(model.model_path),
                    "vram_mb": model.vram_mb,
                    "config": model.config,
                }

        return None

    def _publish_event(self, event_type: EventType, payload: dict[str, Any]) -> None:
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

    async def _cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        if self.worker_process and self.worker_process.is_alive():
            self.logger.info("cleaning_up_worker_process")
            await self._evict_model()

    def stop(self) -> None:
        """Request shutdown."""
        self.logger.info("model_manager_shutdown_requested")
        self._shutdown_requested = True
