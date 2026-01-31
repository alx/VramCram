"""Worker agents for inference execution."""

from vramcram.agents.worker.base import BaseWorker
from vramcram.agents.worker.diffusion_worker import (
    DiffusionWorker,
    diffusion_worker_main,
)
from vramcram.agents.worker.llm_worker import LLMWorker, llm_worker_main

__all__ = [
    "BaseWorker",
    "LLMWorker",
    "llm_worker_main",
    "DiffusionWorker",
    "diffusion_worker_main",
]
