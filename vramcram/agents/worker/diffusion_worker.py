"""Diffusion worker using sd binary subprocess calls."""

import asyncio
import base64
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vramcram.agents.worker.base import BaseWorker
from vramcram.config.models import VramCramConfig
from vramcram.events.bus import EventBus
from vramcram.redis.client import RedisClientFactory
from vramcram.security import validate_binary_path, validate_path, validate_prompt
from vramcram.security.resource_limits import create_resource_limiter


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

        # Validate binary path
        if self.config.security.validate_binary_paths:
            try:
                binary_path = validate_binary_path(self.config.inference.sd_binary_path)
            except ValueError as e:
                raise RuntimeError(f"Invalid sd binary path: {e}")
        else:
            binary_path = Path(self.config.inference.sd_binary_path)
            if not binary_path.exists():
                raise RuntimeError(f"sd binary not found at {binary_path}")

        # Validate model path
        if self.config.security.validate_model_paths:
            try:
                model_path_validated = validate_path(
                    Path(model_path),
                    self.config.security.allowed_model_base_path,
                )
            except ValueError as e:
                raise RuntimeError(f"Invalid model path: {e}")
        else:
            model_path_validated = Path(model_path)
            if not model_path_validated.exists():
                raise RuntimeError(f"Model file not found: {model_path}")

        # Validate optional files (VAE, LLM, LoRA dir)
        if "vae_path" in self.model_config:
            vae_path = self.model_config["vae_path"]
            if self.config.security.validate_model_paths:
                try:
                    validate_path(
                        Path(vae_path),
                        self.config.security.allowed_model_base_path,
                    )
                except ValueError as e:
                    self.logger.warning("vae_validation_failed", path=vae_path, error=str(e))
            elif not os.path.exists(vae_path):
                self.logger.warning("vae_not_found", path=vae_path)

        if "llm_path" in self.model_config:
            llm_path = self.model_config["llm_path"]
            if self.config.security.validate_model_paths:
                try:
                    validate_path(
                        Path(llm_path),
                        self.config.security.allowed_model_base_path,
                    )
                except ValueError as e:
                    self.logger.warning("llm_validation_failed", path=llm_path, error=str(e))
            elif not os.path.exists(llm_path):
                self.logger.warning("llm_not_found", path=llm_path)

        if "lora_model_dir" in self.model_config:
            lora_dir = self.model_config["lora_model_dir"]
            if self.config.security.validate_model_paths:
                try:
                    validate_path(
                        Path(lora_dir),
                        self.config.security.allowed_model_base_path,
                    )
                except ValueError as e:
                    self.logger.warning("lora_validation_failed", path=lora_dir, error=str(e))
            elif not os.path.exists(lora_dir):
                self.logger.warning("lora_dir_not_found", path=lora_dir)

        self.logger.info("sd_binary_validated", binary_path=binary_path, model_path=model_path)

    async def execute_inference(
        self, prompt: str, params: dict[str, Any]
    ) -> str:
        """Execute diffusion inference via sd subprocess.

        Args:
            prompt: Input text prompt for image generation.
            params: Generation parameters (width, height, steps, cfg_scale, etc.).

        Returns:
            String containing base64 data URI or filesystem path (based on config).

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

        # Check config for result format
        result_format = self.config.output.image_result_format

        if result_format == "path":
            # Return filesystem path (backward compatible)
            self.logger.info(
                "image_result_format_path",
                path=str(output_path),
            )
            return str(output_path)

        # Default: return base64
        try:
            with open(output_path, "rb") as f:
                image_bytes = f.read()

            # Encode to base64
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            # Create data URI with MIME type
            image_data_uri = f"data:image/png;base64,{image_base64}"

            self.logger.info(
                "image_encoded_to_base64",
                file_size_bytes=len(image_bytes),
                base64_size=len(image_data_uri),
                path=str(output_path),
            )

            return image_data_uri

        except Exception as e:
            self.logger.error(
                "failed_to_encode_image",
                error=str(e),
                path=str(output_path),
            )
            # Fallback to path
            return str(output_path)

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
        # Validate prompts
        try:
            validate_prompt(prompt, self.config.security.max_prompt_length)
            if negative_prompt:
                validate_prompt(
                    negative_prompt,
                    self.config.security.max_negative_prompt_length,
                )
        except ValueError as e:
            raise RuntimeError(f"Prompt validation failed: {e}")

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

        # Run subprocess with timeout and resource limits
        preexec_fn = create_resource_limiter(
            self.config.security.subprocess_max_memory_mb
        )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=preexec_fn,
        )

        try:
            # Read output with size limits to prevent memory exhaustion
            stdout_chunks: list[bytes] = []
            stderr_chunks: list[bytes] = []

            async def read_with_limit(
                stream: asyncio.StreamReader,
                chunks: list[bytes],
                name: str,
            ) -> None:
                """Read stream with size limit."""
                total_bytes = 0
                max_output = self.config.security.subprocess_max_output_bytes
                while True:
                    chunk = await stream.read(4096)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > max_output:
                        proc.kill()
                        raise RuntimeError(
                            f"Subprocess {name} exceeded output limit of {max_output} bytes"
                        )
                    chunks.append(chunk)

            # Read stdout and stderr concurrently with timeout
            await asyncio.wait_for(
                asyncio.gather(
                    read_with_limit(proc.stdout, stdout_chunks, "stdout"),  # type: ignore
                    read_with_limit(proc.stderr, stderr_chunks, "stderr"),  # type: ignore
                ),
                timeout=self.config.jobs.default_timeout_seconds,
            )

            # Wait for process to complete
            await proc.wait()

            stdout = b"".join(stdout_chunks)
            stderr = b"".join(stderr_chunks)

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
