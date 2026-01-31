# VramCram

VRAM-aware GPU orchestration system for running multiple AI models on a single GPU with automatic model swapping and LRU eviction.

## Features

- **Autonomous Agent Architecture**: Three agent types (Coordinator, Model Manager, Worker) coordinate via Redis events
- **VRAM-Aware Scheduling**: Automatic LRU eviction when GPU memory is full
- **Multi-Model Support**: LLM (llama.cpp) and Diffusion (stable-diffusion.cpp) models
- **Job Queue**: FIFO queue with Redis Streams for reliable job distribution
- **Self-Assignment**: Workers pull jobs autonomously, no central dispatcher bottleneck
- **Health Monitoring**: Heartbeat-based failure detection and recovery
- **RESTful API**: Simple HTTP interface for job submission and status checking

## Dependencies

VramCram requires the following external inference engines:

- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** - Fast LLM inference with llama-server binary
- **[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)** - Efficient diffusion model inference with sd binary
- **Redis 7+** - Message queue and event bus
- **NVIDIA GPU with CUDA** - For GPU-accelerated inference

### Installing Inference Binaries

Download or build the required binaries:

**llama.cpp:**
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
sudo cp build/bin/llama-server /usr/local/bin/
```

**stable-diffusion.cpp:**
```bash
git clone https://github.com/leejet/stable-diffusion.cpp
cd stable-diffusion.cpp
cmake -B build -DSD_CUDA=ON
cmake --build build --config Release
sudo cp build/bin/sd /usr/local/bin/
```

## Architecture

```
┌─────────────┐
│  API Gateway│  (FastAPI, job submission)
└──────┬──────┘
       │ Redis Streams
       ▼
┌─────────────────────┐
│ Coordinator Agent   │  (Job queue, VRAM monitoring, LRU eviction)
└──────┬──────────────┘
       │ Events (Pub/Sub)
       ▼
┌────────────────────────┐
│ Model Manager Agents   │  (Worker lifecycle, one per model)
└──────┬─────────────────┘
       │ Process spawn
       ▼
┌─────────────────────┐
│ Worker Agents       │  (Inference execution, self-assignment)
└─────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Redis 7+
- NVIDIA GPU with CUDA support
- NVIDIA drivers with nvidia-smi

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/vramcram.git
cd vramcram

# Install with uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

### Configuration

```bash
# Copy example config
cp config.example.yaml config.yaml

# Edit config with your model paths and VRAM settings
vim config.yaml
```

Key settings to customize:
- `vram.total_mb`: Your GPU's VRAM capacity
- `models.llm`: Add your GGUF model paths
- `models.diffusion`: Add your Stable Diffusion model paths
- `inference`: Configure paths to llama-server and sd binaries

### Running

```bash
# Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# Set config path
export VRAMCRAM_CONFIG=config.yaml

# Start VramCram
python -m vramcram.main
```

### Usage

Submit an LLM job:
```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-ministral",
    "prompt": "Explain quantum computing in simple terms",
    "params": {"max_tokens": 200}
  }'
# Response: {"job_id": "abc123", "status": "queued"}
```

Check job status:
```bash
curl http://localhost:8000/jobs/abc123
# Response: {"job_id": "abc123", "status": "completed", "model": "llama-ministral", ...}
```

Get result:
```bash
curl http://localhost:8000/jobs/abc123/result
# Response: {"type": "llm", "text": "Quantum computing uses..."}
```

Submit a diffusion job:
```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stable-diffusion-turbo",
    "prompt": "A serene mountain lake at sunset",
    "params": {"width": 512, "height": 512}
  }'
```

Check system health:
```bash
curl http://localhost:8000/health
# Response: {"status": "healthy", "vram_free_mb": 18432, "loaded_models": [...], ...}
```

List available models:
```bash
curl http://localhost:8000/models
```

## Model Setup

VramCram works with models in GGUF format for LLMs and safetensors/GGUF for diffusion models.

### Recommended Models

**LLM Models (GGUF format):**
- [Mistral models](https://huggingface.co/mistralai) - Fast and efficient 7B-8B models
- [Llama 3.2](https://huggingface.co/meta-llama) - Strong performance, 1B-3B models for lower VRAM
- [Qwen 2.5](https://huggingface.co/Qwen) - Excellent multilingual support

Download quantized GGUF versions from [TheBloke](https://huggingface.co/TheBloke) or use [llama.cpp tools](https://github.com/ggerganov/llama.cpp) to quantize models yourself.

**Diffusion Models:**
- [Stable Diffusion XL Turbo](https://huggingface.co/stabilityai/sdxl-turbo) - Fast, high-quality image generation
- [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) - Widely compatible baseline
- Convert to safetensors or GGUF format using [stable-diffusion.cpp tools](https://github.com/leejet/stable-diffusion.cpp)

### Organizing Models

Place models in the `./models/` directory:
```
vramcram/
├── models/
│   ├── ministral-8b-instruct-q4_k_m.gguf
│   ├── sdxl_turbo_fp16.safetensors
│   └── ...
└── config.yaml
```

Update `config.yaml` with the correct model paths and VRAM estimates.

## Example Scripts

VramCram includes example scripts demonstrating common workflows:

- **scripts/simple_workflow_test.py** - Minimal automated test for CI/CD
- **scripts/test_llm_to_image.py** - Interactive LLM-to-image workflow with progress display
- **scripts/demo_api.py** - API usage demonstration

See [scripts/README.md](scripts/README.md) for detailed usage instructions.

## Development

### Running Tests

```bash
# Unit tests
pytest tests/unit -v

# Integration tests (requires Redis)
pytest tests/integration -v

# All tests with coverage
pytest --cov=vramcram --cov-report=html

# Type checking
mypy vramcram

# Linting
ruff check vramcram
```

### Project Structure

```
vramcram/
├── agents/
│   ├── base.py               # BaseAgent abstract class
│   ├── coordinator.py        # Coordinator agent (job queue, eviction)
│   ├── model_manager.py      # Model Manager (worker lifecycle)
│   └── worker/
│       ├── base.py           # Worker base class
│       ├── llm_worker.py     # LLM inference worker
│       └── diffusion_worker.py  # Diffusion inference worker
├── api/
│   ├── gateway.py            # FastAPI endpoints
│   ├── models.py             # Request/response models
│   └── dependencies.py       # Dependency injection
├── config/
│   ├── models.py             # Pydantic config models
│   └── loader.py             # YAML config loader
├── events/
│   ├── bus.py                # EventBus (Redis Pub/Sub wrapper)
│   ├── schema.py             # AgentEvent dataclass
│   └── types.py              # Event type enums
├── gpu/
│   ├── models.py             # VRAMState dataclass
│   └── vram_tracker.py       # VRAM monitoring (pynvml)
├── queue/
│   ├── job.py                # Job dataclass
│   └── registry.py           # ModelRegistry (LRU tracking)
├── redis/
│   └── client.py             # Redis connection factory
└── main.py                   # Application entry point
```

## Configuration Reference

See `config.example.yaml` for full configuration options.

### VRAM Settings

- `vram.total_mb`: Total GPU VRAM (get with `nvidia-smi`)
- `vram.safety_margin_mb`: Reserved memory for system overhead
- `vram.monitoring_interval_seconds`: How often to check VRAM usage

### Model Configuration

Each model requires:
- `name`: Unique identifier
- `type`: `llm` or `diffusion`
- `model_path`: Absolute path to model file
- `vram_mb`: Estimated VRAM usage when loaded
- `config`: Model-specific parameters

### Agent Settings

- `heartbeat_interval_seconds`: How often agents send heartbeats (default: 10s)
- `heartbeat_timeout_seconds`: When to mark agent as failed (default: 30s)
- `worker_ready_timeout_seconds`: Timeout for worker startup (default: 60s)
- `graceful_shutdown_timeout_seconds`: Time to drain jobs on shutdown (default: 300s)

## How It Works

### Job Lifecycle

1. **Submission**: Client POSTs to `/jobs` endpoint
2. **Queuing**: Job added to Redis Stream, status set to "queued"
3. **Coordination**: Coordinator polls queue, checks if model is loaded
4. **Loading**: If model not loaded, Coordinator requests Model Manager to load it
5. **Eviction**: If VRAM insufficient, Coordinator evicts LRU model first
6. **Assignment**: Worker pulls job from stream via XREADGROUP (self-assignment)
7. **Execution**: Worker runs inference, stores result in Redis
8. **Completion**: Worker publishes job.completed event, ACKs message
9. **Retrieval**: Client GETs result from `/jobs/{id}/result`

### VRAM Management

- Coordinator monitors VRAM every 5 seconds via pynvml
- Each model has estimated `vram_mb` in config
- When VRAM insufficient for new model:
  1. Select LRU victim (oldest `last_used` timestamp)
  2. Send eviction request to Model Manager
  3. Model Manager gracefully shuts down worker (SIGTERM → SIGKILL)
  4. Load new model after eviction completes

### Failure Recovery

- All agents send heartbeats every 10 seconds
- Coordinator detects missing heartbeats (30s timeout)
- On worker failure:
  - Job marked as "failed"
  - Model marked as unloaded
  - New worker spawned on next job request
- On Coordinator failure: System requires manual restart (singleton agent)

## Deployment

### Systemd Service

```bash
# Install service
sudo cp deploy/systemd/vramcram.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable vramcram
sudo systemctl start vramcram

# Check status
sudo systemctl status vramcram

# View logs
sudo journalctl -u vramcram -f
```

### Production Considerations

- Run Redis with persistence (AOF or RDB)
- Set up monitoring with Prometheus (metrics on `:9090/metrics`)
- Use reverse proxy (nginx) for API gateway
- Configure log rotation for structlog output
- Set resource limits in systemd unit
- Use dedicated `vramcram` user (non-root)

## Troubleshooting

### Worker fails to load model

Check:
- Model path exists and is readable
- VRAM estimate in config is accurate
- GPU has sufficient free VRAM (`nvidia-smi`)

### Jobs stuck in "queued" status

Check:
- Coordinator is running (heartbeat in logs)
- Model Manager spawned for the model
- Worker published "ready" event within 60s

### VRAM not being released

Check:
- Worker processes fully terminated (`ps aux | grep vramcram`)
- No zombie processes
- pynvml correctly detecting VRAM release

### Redis connection errors

Check:
- Redis is running (`redis-cli ping`)
- Connection settings in config.yaml
- Firewall rules if Redis is remote

## License

MIT

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Submit a pull request

## Roadmap

- [ ] Multi-GPU support
- [ ] Priority queues
- [ ] Job cancellation
- [ ] Streaming responses
- [ ] WebSocket API
- [ ] Model preloading hints
- [ ] Advanced scheduling policies (round-robin, weighted)
- [ ] Job batching for throughput optimization
