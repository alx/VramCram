"""Main entry point for VramCram system."""

import asyncio
import signal
import sys
from pathlib import Path

import structlog
import uvicorn

from vramcram.agents.coordinator import CoordinatorAgent
from vramcram.agents.model_manager import ModelManagerAgent
from vramcram.api.gateway import create_app
from vramcram.config.loader import load_config
from vramcram.events.bus import EventBus
from vramcram.gpu.vram_tracker import VRAMTracker
from vramcram.redis.client import RedisClientFactory

logger = structlog.get_logger()


class VramCramSystem:
    """Main VramCram system orchestrator.

    Responsible for:
    - Loading configuration
    - Initializing Redis factory, EventBus, VRAMTracker
    - Starting Coordinator agent
    - Starting Model Manager agents (one per model in config)
    - Starting API server with uvicorn
    - Running all with asyncio.gather()
    """

    def __init__(self, config_path: Path) -> None:
        """Initialize VramCram system.

        Args:
            config_path: Path to configuration file.
        """
        self.config_path = config_path
        self.config = load_config(config_path)
        self.logger = logger.bind(component="vramcram_system")

        # Initialize core components
        self.redis_factory = RedisClientFactory(self.config.redis)
        self.redis_client = self.redis_factory.create_client()
        self.event_bus = EventBus(self.redis_client)
        self.vram_tracker = VRAMTracker(gpu_index=0)

        # Agents
        self.coordinator: CoordinatorAgent | None = None
        self.model_managers: list[ModelManagerAgent] = []

        # API app
        self.api_app = create_app(self.config, self.redis_factory, self.event_bus)

        # Shutdown flag
        self._shutdown_requested = False

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum: int, frame: object) -> None:
        """Handle shutdown signals.

        Args:
            signum: Signal number.
            frame: Current stack frame.
        """
        self.logger.info("received_shutdown_signal", signal=signum)
        self._shutdown_requested = True

    async def run(self) -> None:
        """Run VramCram system."""
        self.logger.info("vramcram_starting", config_path=str(self.config_path))

        try:
            # Create Coordinator with its own EventBus instance
            coordinator_bus = EventBus(self.redis_factory.create_client())
            self.coordinator = CoordinatorAgent(
                agent_id="coordinator",
                event_bus=coordinator_bus,
                config=self.config,
                vram_tracker=self.vram_tracker,
            )

            # Create Model Managers (one per model), each with its own EventBus
            for model in self.config.models.llm:
                manager_bus = EventBus(self.redis_factory.create_client())
                manager = ModelManagerAgent(
                    agent_id=f"model-manager-{model.name}",
                    model_name=model.name,
                    event_bus=manager_bus,
                    config=self.config,
                )
                self.model_managers.append(manager)

            for model in self.config.models.diffusion:
                manager_bus = EventBus(self.redis_factory.create_client())
                manager = ModelManagerAgent(
                    agent_id=f"model-manager-{model.name}",
                    model_name=model.name,
                    event_bus=manager_bus,
                    config=self.config,
                )
                self.model_managers.append(manager)

            self.logger.info(
                "agents_created",
                coordinator=True,
                model_managers=len(self.model_managers),
            )

            # Start all agents and API server
            tasks = [
                asyncio.create_task(self.coordinator.run()),
                *[asyncio.create_task(manager.run()) for manager in self.model_managers],
                asyncio.create_task(self._run_api_server()),
            ]

            # Run until shutdown
            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception:
            self.logger.exception("vramcram_error")
            raise
        finally:
            await self._cleanup()
            self.logger.info("vramcram_stopped")

    async def _run_api_server(self) -> None:
        """Run API server with uvicorn."""
        config = uvicorn.Config(
            self.api_app,
            host=self.config.api.host,
            port=self.config.api.port,
            log_level="info",
        )
        server = uvicorn.Server(config)

        self.logger.info(
            "api_server_starting",
            host=self.config.api.host,
            port=self.config.api.port,
        )

        await server.serve()

    async def _cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        self.logger.info("cleaning_up")

        # Stop coordinator
        if self.coordinator:
            self.coordinator.stop()

        # Stop model managers
        for manager in self.model_managers:
            manager.stop()

        # Close Redis connections
        self.redis_factory.close()


async def main(config_path: Path | None = None) -> None:
    """Main entry point.

    Args:
        config_path: Path to configuration file. Defaults to config.yaml in current directory.
    """
    if config_path is None:
        config_path = Path("config.yaml")

    if not config_path.exists():
        logger.error("config_not_found", config_path=str(config_path))
        sys.exit(1)

    system = VramCramSystem(config_path)
    await system.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VramCram GPU Orchestration System")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration file",
    )

    args = parser.parse_args()

    asyncio.run(main(args.config))
