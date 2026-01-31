# VramCram Test Scripts

This directory contains test scripts for validating VramCram functionality.

## Prerequisites

1. **VramCram must be running:**
   ```bash
   uv run python -m vramcram.main --config config.yaml
   ```

2. **Redis must be running:**
   ```bash
   docker run -d -p 6379:6379 redis:7-alpine
   ```

3. **Models must be configured** in `config.yaml` with valid paths

## Test Scripts

### 1. LLM-to-Image Workflow Test (Fancy)

**File:** `test_llm_to_image.py`

**Description:** Interactive test with colored output demonstrating the full workflow:
- Generate text description using an LLM
- Use that description to generate an image

**Usage:**
```bash
# Basic usage (auto-detects models)
uv run python scripts/test_llm_to_image.py

# Specify models
uv run python scripts/test_llm_to_image.py \
    --llm-model llama-ministral \
    --diffusion-model stable-diffusion-turbo

# Custom subject
uv run python scripts/test_llm_to_image.py \
    --subject "a futuristic cityscape at night"

# All options
uv run python scripts/test_llm_to_image.py \
    --host localhost \
    --port 8000 \
    --llm-model llama-ministral \
    --diffusion-model stable-diffusion-turbo \
    --subject "a serene forest with autumn colors" \
    --max-tokens 150 \
    --timeout 300
```

**Options:**
- `--host`: API host (default: localhost)
- `--port`: API port (default: 8000)
- `--llm-model`: LLM model name (auto-detected if not specified)
- `--diffusion-model`: Diffusion model name (auto-detected if not specified)
- `--subject`: Subject for the LLM to describe (default: "a serene mountain landscape at sunset")
- `--max-tokens`: Max tokens for LLM (default: 150)
- `--timeout`: Timeout per job in seconds (default: 300)

**Output:**
- Colored terminal output with progress indicators
- Shows system health, model availability
- Displays generated text description
- Reports image output path and file size
- Summary with timing information

### 2. Simple Workflow Test (Minimal)

**File:** `simple_workflow_test.py`

**Description:** Minimal automated test suitable for CI/CD pipelines:
- No dependencies beyond requests
- Simple pass/fail output
- Fixed test prompt
- Exit code 0 on success, 1 on failure

**Usage:**
```bash
# Run test
uv run python scripts/simple_workflow_test.py

# Check exit code
echo $?
```

**Use Cases:**
- Automated testing
- CI/CD pipelines
- Quick smoke tests
- Integration validation

## Example Workflow

### Full Manual Test

1. **Start Redis:**
   ```bash
   docker run -d -p 6379:6379 --name vramcram-redis redis:7-alpine
   ```

2. **Configure VramCram:**
   ```bash
   cp config.example.yaml config.yaml
   # Edit config.yaml with your model paths
   ```

3. **Start VramCram:**
   ```bash
   uv run python -m vramcram.main --config config.yaml
   ```

4. **Run test (in another terminal):**
   ```bash
   uv run python scripts/test_llm_to_image.py \
       --subject "a majestic dragon flying over a medieval castle"
   ```

5. **View results:**
   - Check terminal output for generated description
   - Find generated image at path shown in output (typically `./outputs/YYYY-MM-DD/job_*.png`)

### Quick Automated Test

```bash
# Start services
docker run -d -p 6379:6379 redis:7-alpine
uv run python -m vramcram.main --config config.yaml &

# Wait for startup
sleep 5

# Run test
uv run python scripts/simple_workflow_test.py

# Check result
if [ $? -eq 0 ]; then
    echo "✓ Workflow test passed"
else
    echo "✗ Workflow test failed"
    exit 1
fi
```

## Expected Behavior

### Successful Run

1. **Health Check**: System reports healthy with models available
2. **Model Detection**: At least one LLM and one diffusion model found
3. **LLM Job**:
   - Submits successfully
   - Completes within timeout (typically 10-60 seconds)
   - Returns descriptive text (50-200 characters)
4. **Image Job**:
   - Submits with LLM output as prompt
   - Completes within timeout (typically 30-120 seconds)
   - Creates PNG file at specified path
5. **Total Time**: Usually 40-180 seconds depending on hardware

### Common Issues

**Connection Refused:**
```
ERROR: Health check failed: Connection refused
```
- VramCram is not running
- Wrong host/port specified

**No Models Available:**
```
ERROR: No LLM models available!
```
- Check config.yaml model paths
- Ensure model files exist
- Verify model type is correct (llm/diffusion)

**Job Timeout:**
```
ERROR: LLM job timeout
```
- Model loading is slow (increase --timeout)
- GPU is busy with other tasks
- Model file is corrupted or incompatible
- Check VramCram logs for errors

**Image File Not Found:**
```
ERROR: Image file not found
```
- Check output directory permissions
- Ensure ./outputs directory exists and is writable
- May need to create directory: `mkdir -p ./outputs`

## Customization

### Creating Your Own Test

```python
import requests
import json
import time

# 1. Submit LLM job
response = requests.post(
    "http://localhost:8000/jobs",
    json={
        "model": "your-llm-model",
        "prompt": "Your prompt here",
        "params": {"max_tokens": 100}
    }
)
llm_job_id = response.json()["job_id"]

# 2. Wait for completion
while True:
    status = requests.get(f"http://localhost:8000/jobs/{llm_job_id}").json()
    if status["status"] == "completed":
        break
    time.sleep(2)

# 3. Get result
result = requests.get(f"http://localhost:8000/jobs/{llm_job_id}/result").json()
text = json.loads(result["result"])["text"]

# 4. Use text for image generation
response = requests.post(
    "http://localhost:8000/jobs",
    json={
        "model": "your-diffusion-model",
        "prompt": text,
        "params": {"width": 512, "height": 512}
    }
)
image_job_id = response.json()["job_id"]

# 5. Wait and get image path
# ... (similar to step 2-3)
```

## Performance Benchmarks

Typical execution times on different hardware:

| Hardware | LLM Time | Image Time | Total |
|----------|----------|------------|-------|
| RTX 4090 | 5-15s    | 10-30s     | 15-45s |
| RTX 3080 | 10-30s   | 20-60s     | 30-90s |
| RTX 2080 | 20-60s   | 40-120s    | 60-180s |

*Times vary based on model size, parameters, and system load*

## Troubleshooting

### Enable Debug Logging

In `config.yaml`:
```yaml
logging:
  level: DEBUG
  format: json
```

### Check VramCram Logs

```bash
# If running in foreground, check terminal output

# If using systemd:
journalctl -u vramcram -f

# Check Redis for events:
redis-cli MONITOR | grep events:
```

### Manual API Testing

```bash
# Check health
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Submit job
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-ministral", "prompt": "Hello", "params": {}}'

# Check status
curl http://localhost:8000/jobs/{job_id}
```

## Contributing

To add a new test script:

1. Create script in `scripts/` directory
2. Make it executable: `chmod +x scripts/your_script.py`
3. Add shebang: `#!/usr/bin/env python3`
4. Document usage in this README
5. Include error handling and clear output
6. Return proper exit codes (0 = success, 1 = failure)
