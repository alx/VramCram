"""LLM worker using llama-server HTTP API."""

import os
from typing import Any

import httpx

from vramcram.agents.worker.base import BaseWorker
from vramcram.config.models import VramCramConfig
from vramcram.events.bus import EventBus
from vramcram.redis.client import RedisClientFactory
from vramcram.security import validate_prompt


class LLMWorker(BaseWorker):
    """Worker for LLM inference using llama-server HTTP API.

    Connects to llama-server spawned by Model Manager and executes text generation via HTTP.
    Implements the abstract methods from BaseWorker.

    Pattern:
    - Validate llama-server is running and accessible
    - Execute inference via HTTP POST to /v1/chat/completions
    - llama-server handles VRAM allocation adaptively
    """

    def __init__(
        self,
        worker_id: str,
        model_name: str,
        event_bus: EventBus,
        redis_factory: RedisClientFactory,
        config: VramCramConfig,
    ) -> None:
        """Initialize LLM worker.

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

    def _get_model_index(self) -> int:
        """Get index of this model in LLM models list.

        Returns:
            Model index (0-based).
        """
        for i, model in enumerate(self.config.models.llm):
            if model.name == self.model_name:
                return i
        return 0

    async def load_inference_model(self) -> None:
        """Validate llama-server is running and accessible."""
        # Get model config
        model_config = self._get_model_config()
        if not model_config:
            raise ValueError(f"Model config not found: {self.model_name}")

        self.model_config = model_config.get("config", {})

        # Calculate port based on model index
        model_index = self._get_model_index()
        port = 8081 + model_index
        server_url = f"http://127.0.0.1:{port}"

        # Check server is running
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{server_url}/health")
                response.raise_for_status()
                self.logger.info(
                    "llama_server_validated",
                    server_url=server_url,
                    model_name=self.model_name
                )
        except Exception as e:
            raise RuntimeError(
                f"llama-server not accessible at {server_url}: {str(e)}"
            )

    async def execute_inference(
        self, prompt: str, params: dict[str, Any]
    ) -> str:
        """Execute LLM inference via llama-server HTTP API.

        Args:
            prompt: Input text prompt.
            params: Generation parameters (max_tokens, temperature, etc.).

        Returns:
            Direct text string containing generated text.

        Raises:
            RuntimeError: If inference fails.
        """
        self.logger.debug("executing_inference", prompt_length=len(prompt))

        # Extract parameters with defaults
        max_tokens = params.get("max_tokens", self.model_config.get("max_tokens", 512))
        temperature = params.get(
            "temperature", self.model_config.get("temperature", 0.7)
        )
        top_p = params.get("top_p", self.model_config.get("top_p", 0.95))
        top_k = params.get("top_k", self.model_config.get("top_k", 40))

        # Run inference
        result = await self._run_llama_server_inference(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        self.logger.debug(
            "inference_completed",
            generated_length=len(result),
        )

        return result

    async def _run_llama_server_inference(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
    ) -> str:
        """Call llama-server HTTP API for inference.

        Uses OpenAI-compatible /v1/chat/completions endpoint.

        Args:
            prompt: Input text prompt.
            max_tokens: Number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.

        Returns:
            Generated text.

        Raises:
            RuntimeError: If HTTP request fails or times out.
        """
        # Validate prompt
        try:
            validate_prompt(prompt, self.config.security.max_prompt_length)
        except ValueError as e:
            raise RuntimeError(f"Prompt validation failed: {e}")

        # Calculate port based on model index
        model_index = self._get_model_index()
        port = 8081 + model_index
        server_url = f"http://127.0.0.1:{port}"

        # Build chat completion request
        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }

        # Make HTTP request with timeout
        async with httpx.AsyncClient(
            timeout=self.config.jobs.default_timeout_seconds
        ) as client:
            try:
                response = await client.post(
                    f"{server_url}/v1/chat/completions",
                    json=payload,
                )
                response.raise_for_status()

                # Check response size before processing
                content_length = len(response.content)
                max_output = self.config.security.subprocess_max_output_bytes
                if content_length > max_output:
                    raise RuntimeError(
                        f"Response size {content_length} exceeds limit of {max_output} bytes"
                    )

                result = response.json()
                generated_text = result["choices"][0]["message"]["content"]

                # Truncate if generated text is too large
                if len(generated_text.encode()) > max_output:
                    self.logger.warning(
                        "response_truncated",
                        original_size=len(generated_text),
                        max_size=max_output,
                    )
                    # Truncate to safe size
                    generated_text = generated_text[:max_output // 2]  # Conservative estimate

                return generated_text

            except httpx.TimeoutException:
                raise RuntimeError(
                    f"LLM inference timeout after {self.config.jobs.default_timeout_seconds}s"
                )
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"llama-server request failed: {e.response.text}")
            except Exception as e:
                raise RuntimeError(f"llama-server communication error: {str(e)}")

    def _get_model_config(self) -> dict[str, Any] | None:
        """Get configuration for this model.

        Returns:
            Model config dict or None if not found.
        """
        for model in self.config.models.llm:
            if model.name == self.model_name:
                return {
                    "type": "llm",
                    "model_path": str(model.model_path),
                    "vram_mb": model.vram_mb,
                    "config": model.config,
                }
        return None


def llm_worker_main(model_name: str, config: VramCramConfig) -> None:
    """Main entry point for LLM worker process.

    This function is called by multiprocessing.Process in the Model Manager.

    Args:
        model_name: Name of model to load.
        config: System configuration.
    """
    import asyncio
    import signal

    # Create worker ID
    worker_id = f"llm-worker-{model_name}-{os.getpid()}"

    # Initialize Redis and EventBus
    redis_factory = RedisClientFactory(config.redis)
    redis_client = redis_factory.create_client()
    event_bus = EventBus(redis_client)

    # Create worker
    worker = LLMWorker(
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
