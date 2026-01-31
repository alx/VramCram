"""Diffusion worker using sd binary subprocess calls."""

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vramcram.agents.worker.base import BaseWorker
from vramcram.config.models import VramCramConfig
from vramcram.events.bus import EventBus
from vramcram.redis.client import RedisClientFactory


class DiffusionWorker(BaseWorker):
    """Worker for diffusion inference using sd binary subprocess calls.

    Validates sd binary and executes image generation via subprocess.
    Implements the abstract methods from BaseWorker.

    Output:
    - Save images to configured output path/{date}/job_{timestamp}.png
    - Return image path in result
    """

    def __init__(
        self,
        worker_id: str,
        model_name: str,
        event_bus: EventBus,
        redis_factory: RedisClientFactory,
        config: VramCramConfig,
    ) -> None:
        """Initialize diffusion worker.

        Args:
            worker_id: Unique worker identifier.
            model_name: Name of model this worker handles.
            event_bus: Event bus for communication.
            redis_factory: Redis client factory.
            config: System configuration.
        """
        super().__init__(worker_id, model_name, event_bus, redis_factory, config)
        self.model: Any = None
        self.model_config: dict[str, Any] = {}

    async def load_inference_model(self) -> None:
        """Validate sd binary and model files."""
        # Get model config
        model_config = self._get_model_config()
        if not model_config:
            raise ValueError(f"Model config not found: {self.model_name}")

        self.model_config = model_config.get("config", {})
        model_path = model_config.get("model_path")

        if not model_path:
            raise ValueError(f"Model path not found for: {self.model_name}")

        # Check binary exists
        binary_path = self.config.inference.sd_binary_path
        if not os.path.exists(binary_path):
            raise RuntimeError(f"sd binary not found at {binary_path}")

        # Check model file exists
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model file not found: {model_path}")

        # Check optional files (VAE, LLM, LoRA dir)
        if "vae_path" in self.model_config:
            vae_path = self.model_config["vae_path"]
            if not os.path.exists(vae_path):
                self.logger.warning("vae_not_found", path=vae_path)

        if "llm_path" in self.model_config:
            llm_path = self.model_config["llm_path"]
            if not os.path.exists(llm_path):
                self.logger.warning("llm_not_found", path=llm_path)

        if "lora_model_dir" in self.model_config:
            lora_dir = self.model_config["lora_model_dir"]
            if not os.path.exists(lora_dir):
                self.logger.warning("lora_dir_not_found", path=lora_dir)

        self.logger.info("sd_binary_validated", binary_path=binary_path, model_path=model_path)

    async def execute_inference(
        self, prompt: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute diffusion inference via sd subprocess.

        Args:
            prompt: Input text prompt for image generation.
            params: Generation parameters (width, height, steps, cfg_scale, etc.).

        Returns:
            Dictionary with "image_path" key containing path to saved image.

        Raises:
            RuntimeError: If inference fails.
        """
        self.logger.debug("executing_inference", prompt=prompt)

        # Extract parameters with defaults
        negative_prompt = params.get("negative_prompt", "")
        width = params.get("width", self.model_config.get("width", 512))
        height = params.get("height", self.model_config.get("height", 512))
        sample_steps = params.get(
            "sample_steps", self.model_config.get("sample_steps", 4)
        )
        cfg_scale = params.get("cfg_scale", self.model_config.get("cfg_scale", 1.0))

        # Generate output path
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        output_dir = Path(self.config.output.base_path) / date_str
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"job_{timestamp}.png"

        # Run inference
        await self._run_sd_inference(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            sample_steps=sample_steps,
            cfg_scale=cfg_scale,
            output_path=output_path,
        )

        self.logger.info("image_saved", output_path=str(output_path))

        return {"image_path": str(output_path)}

    async def _run_sd_inference(
        self,
        prompt: str,
        output_path: Path,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        sample_steps: int = 4,
        cfg_scale: float = 1.0,
    ) -> None:
        """Run sd subprocess to generate image.

        Command format:
        sd -m <model> -p "<prompt>" -n "<neg_prompt>" -H <height> -W <width>
           --steps <steps> --cfg-scale <cfg> -o <output>
           [--vae <vae>] [--llm <llm>] [--lora-model-dir <dir>]

        Args:
            prompt: Input text prompt.
            output_path: Path to save generated image.
            negative_prompt: Negative prompt.
            width: Image width.
            height: Image height.
            sample_steps: Number of diffusion steps.
            cfg_scale: CFG guidance scale.

        Raises:
            RuntimeError: If subprocess fails or times out.
        """
        model_config = self._get_model_config()
        if not model_config:
            raise RuntimeError(f"Model config not found: {self.model_name}")

        cmd = [
            self.config.inference.sd_binary_path,
            "--diffusion-model", model_config["model_path"],
            "-p", prompt,
            "-n", negative_prompt,
            "-H", str(height),
            "-W", str(width),
            "--steps", str(sample_steps),
            "--cfg-scale", str(cfg_scale),
            "-o", str(output_path),
        ]

        # Add optional parameters
        if "vae_path" in self.model_config:
            vae_path = self.model_config["vae_path"]
            if os.path.exists(vae_path):
                cmd.extend(["--vae", vae_path])
                # Enable VAE optimization when VAE is used
                cmd.append("--vae-conv-direct")

        if "llm_path" in self.model_config:
            llm_path = self.model_config["llm_path"]
            if os.path.exists(llm_path):
                cmd.extend(["--llm", llm_path])

        if "lora_model_dir" in self.model_config:
            lora_dir = self.model_config["lora_model_dir"]
            if os.path.exists(lora_dir):
                cmd.extend(["--lora-model-dir", lora_dir])

        # Memory optimization flags for 8GB GPU
        # These flags reduce VRAM usage by moving components to CPU
        # and using optimized kernels
        cmd.extend([
            "--clip-on-cpu",           # Move CLIP text encoder to CPU (saves 1-2GB)
            "--diffusion-fa",          # Flash Attention (reduces memory)
            "--diffusion-conv-direct", # Direct convolutions (reduces memory)
            "--rng", "cpu",            # CPU-based random number generation
            "-v",                      # Verbose output for debugging
        ])

        # Run subprocess with timeout
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.jobs.default_timeout_seconds,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(
                f"SD inference timeout after {self.config.jobs.default_timeout_seconds}s"
            )

        if proc.returncode != 0:
            raise RuntimeError(f"sd binary failed: {stderr.decode()}")

        # Verify output file was created
        if not output_path.exists():
            raise RuntimeError(f"SD did not create output file: {output_path}")

    def _get_model_config(self) -> dict[str, Any] | None:
        """Get configuration for this model.

        Returns:
            Model config dict or None if not found.
        """
        for model in self.config.models.diffusion:
            if model.name == self.model_name:
                return {
                    "type": "diffusion",
                    "model_path": str(model.model_path),
                    "vram_mb": model.vram_mb,
                    "config": model.config,
                }
        return None


def diffusion_worker_main(model_name: str, config: VramCramConfig) -> None:
    """Main entry point for diffusion worker process.

    This function is called by multiprocessing.Process in the Model Manager.

    Args:
        model_name: Name of model to load.
        config: System configuration.
    """
    import asyncio
    import signal

    # Create worker ID
    worker_id = f"diffusion-worker-{model_name}-{os.getpid()}"

    # Initialize Redis and EventBus
    redis_factory = RedisClientFactory(config.redis)
    redis_client = redis_factory.create_client()
    event_bus = EventBus(redis_client)

    # Create worker
    worker = DiffusionWorker(
        worker_id=worker_id,
        model_name=model_name,
        event_bus=event_bus,
        redis_factory=redis_factory,
        config=config,
    )

    # Set up signal handler for graceful shutdown
    def signal_handler(_signum, _frame):
        worker.stop()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Run worker
    try:
        asyncio.run(worker.run())
    except (KeyboardInterrupt, SystemExit):
        pass  # Already handled by signal handler
