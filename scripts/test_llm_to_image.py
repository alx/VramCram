#!/usr/bin/env python3
"""Test script for LLM-to-Image workflow.

This script demonstrates the full VramCram workflow:
1. Generate an image description using an LLM
2. Use that description to generate an image with a diffusion model

Usage:
    python scripts/test_llm_to_image.py --host localhost --port 8000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str) -> None:
    """Print a colored header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 70}{Colors.END}\n")


def print_step(step: int, text: str) -> None:
    """Print a step indicator."""
    print(f"{Colors.BOLD}{Colors.CYAN}[Step {step}]{Colors.END} {text}")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}✓{Colors.END} {text}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}✗{Colors.END} {text}")


def print_info(text: str) -> None:
    """Print an info message."""
    print(f"{Colors.BLUE}ℹ{Colors.END} {text}")


def check_health(base_url: str) -> bool:
    """Check if VramCram is healthy.

    Args:
        base_url: Base URL of VramCram API.

    Returns:
        True if healthy, False otherwise.
    """
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        response.raise_for_status()
        data = response.json()

        if data["status"] == "healthy":
            print_success(f"System healthy - {data['models_available']} models available")
            print_info(f"Redis: {'✓' if data['redis_connected'] else '✗'}")
            print_info(f"Jobs in queue: {data['jobs_queued']}")
            return True
        else:
            print_error(f"System unhealthy: {data}")
            return False
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False


def list_models(base_url: str) -> dict[str, list[str]]:
    """List available models.

    Args:
        base_url: Base URL of VramCram API.

    Returns:
        Dictionary with 'llm' and 'diffusion' lists of model names.
    """
    try:
        response = requests.get(f"{base_url}/models", timeout=5)
        response.raise_for_status()
        models = response.json()

        llm_models = [m["name"] for m in models if m["type"] == "llm"]
        diffusion_models = [m["name"] for m in models if m["type"] == "diffusion"]

        print_success(f"Found {len(llm_models)} LLM models: {', '.join(llm_models)}")
        print_success(
            f"Found {len(diffusion_models)} diffusion models: {', '.join(diffusion_models)}"
        )

        return {"llm": llm_models, "diffusion": diffusion_models}
    except Exception as e:
        print_error(f"Failed to list models: {e}")
        return {"llm": [], "diffusion": []}


def submit_job(
    base_url: str, model: str, prompt: str, params: dict | None = None
) -> str | None:
    """Submit a job to VramCram.

    Args:
        base_url: Base URL of VramCram API.
        model: Model name to use.
        prompt: Input prompt.
        params: Optional model parameters.

    Returns:
        Job ID if successful, None otherwise.
    """
    if params is None:
        params = {}

    try:
        response = requests.post(
            f"{base_url}/jobs",
            json={"model": model, "prompt": prompt, "params": params},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        job_id = data["job_id"]
        print_success(f"Job submitted: {job_id}")
        print_info(f"Model: {model}")
        print_info(f"Status: {data['status']}")

        return job_id
    except Exception as e:
        print_error(f"Failed to submit job: {e}")
        return None


def wait_for_job(
    base_url: str, job_id: str, timeout: int = 300, poll_interval: int = 2
) -> dict | None:
    """Wait for a job to complete.

    Args:
        base_url: Base URL of VramCram API.
        job_id: Job ID to wait for.
        timeout: Maximum time to wait in seconds.
        poll_interval: Seconds between status checks.

    Returns:
        Job status dict if completed, None if timeout or error.
    """
    start_time = time.time()

    print_info(f"Waiting for job {job_id} to complete...")

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{base_url}/jobs/{job_id}", timeout=5)
            response.raise_for_status()
            data = response.json()

            status = data["status"]

            # Print progress
            elapsed = int(time.time() - start_time)
            print(
                f"\r{Colors.YELLOW}⏳{Colors.END} Status: {status:12s} | "
                f"Elapsed: {elapsed}s",
                end="",
                flush=True,
            )

            if status == "completed":
                print()  # New line after progress
                duration = data.get("duration_ms", 0)
                print_success(f"Job completed in {duration}ms")
                return data
            elif status == "failed":
                print()  # New line after progress
                error = data.get("error", "Unknown error")
                print_error(f"Job failed: {error}")
                return None

            time.sleep(poll_interval)
        except Exception as e:
            print()  # New line after progress
            print_error(f"Error checking job status: {e}")
            time.sleep(poll_interval)

    print()  # New line after progress
    print_error(f"Job timed out after {timeout}s")
    return None


def get_job_result(base_url: str, job_id: str) -> dict | None:
    """Get job result.

    Args:
        base_url: Base URL of VramCram API.
        job_id: Job ID.

    Returns:
        Result dict or None if error.
    """
    try:
        response = requests.get(f"{base_url}/jobs/{job_id}/result", timeout=5)
        response.raise_for_status()
        data = response.json()

        if data["status"] == "completed" and data["result"]:
            return json.loads(data["result"])
        else:
            print_error(f"Job not completed or no result: {data['status']}")
            return None
    except Exception as e:
        print_error(f"Failed to get job result: {e}")
        return None


def main() -> int:
    """Run the LLM-to-Image workflow test."""
    parser = argparse.ArgumentParser(
        description="Test LLM-to-Image workflow with VramCram"
    )
    parser.add_argument("--host", default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument(
        "--llm-model",
        help="LLM model name (auto-detected if not specified)",
    )
    parser.add_argument(
        "--diffusion-model",
        help="Diffusion model name (auto-detected if not specified)",
    )
    parser.add_argument(
        "--subject",
        default="a serene mountain landscape at sunset",
        help="Subject for the LLM to describe",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="Max tokens for LLM generation",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for each job in seconds",
    )

    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    print_header("VramCram LLM-to-Image Workflow Test")

    # Step 1: Health check
    print_step(1, "Checking system health...")
    if not check_health(base_url):
        return 1

    # Step 2: List models
    print_step(2, "Listing available models...")
    models = list_models(base_url)

    if not models["llm"]:
        print_error("No LLM models available!")
        return 1

    if not models["diffusion"]:
        print_error("No diffusion models available!")
        return 1

    # Select models
    llm_model = args.llm_model or models["llm"][0]
    diffusion_model = args.diffusion_model or models["diffusion"][0]

    print_info(f"Selected LLM: {llm_model}")
    print_info(f"Selected Diffusion: {diffusion_model}")

    # Step 3: Generate image description with LLM
    print_step(3, "Generating image description with LLM...")

    llm_prompt = f"""Create a detailed, vivid description for an image of {args.subject}.
Include specific details about colors, lighting, composition, and mood.
Keep it concise but descriptive (2-3 sentences).
Description:"""

    print_info(f"LLM Prompt: {llm_prompt[:100]}...")

    llm_job_id = submit_job(
        base_url,
        llm_model,
        llm_prompt,
        params={"max_tokens": args.max_tokens, "temperature": 0.8},
    )

    if not llm_job_id:
        return 1

    # Step 4: Wait for LLM completion
    print_step(4, "Waiting for LLM to generate description...")
    llm_status = wait_for_job(base_url, llm_job_id, timeout=args.timeout)

    if not llm_status:
        return 1

    # Step 5: Get LLM result
    print_step(5, "Retrieving generated description...")
    llm_result = get_job_result(base_url, llm_job_id)

    if not llm_result or "text" not in llm_result:
        print_error("Failed to get LLM result")
        return 1

    description = llm_result["text"].strip()
    print_success("Generated description:")
    print(f"{Colors.BOLD}{Colors.CYAN}{description}{Colors.END}\n")

    # Step 6: Generate image from description
    print_step(6, "Generating image from description...")

    # Use the LLM-generated description as the diffusion prompt
    image_prompt = description

    print_info(f"Image Prompt: {image_prompt[:100]}...")

    image_job_id = submit_job(
        base_url,
        diffusion_model,
        image_prompt,
        params={
            "width": 512,
            "height": 512,
            "sample_steps": 4,
            "cfg_scale": 7.0,
        },
    )

    if not image_job_id:
        return 1

    # Step 7: Wait for image generation
    print_step(7, "Waiting for image generation...")
    image_status = wait_for_job(base_url, image_job_id, timeout=args.timeout)

    if not image_status:
        return 1

    # Step 8: Get image result
    print_step(8, "Retrieving generated image...")
    image_result = get_job_result(base_url, image_job_id)

    if not image_result or "image_path" not in image_result:
        print_error("Failed to get image result")
        return 1

    image_path = image_result["image_path"]
    print_success(f"Image saved to: {image_path}")

    # Verify image exists
    if Path(image_path).exists():
        size = Path(image_path).stat().st_size
        print_info(f"Image size: {size:,} bytes")
    else:
        print_error(f"Image file not found at {image_path}")
        return 1

    # Summary
    print_header("Workflow Complete!")
    print(f"{Colors.GREEN}✓ LLM Job:{Colors.END} {llm_job_id}")
    print(f"  Duration: {llm_status['duration_ms']}ms")
    print(f"  Generated: {len(description)} characters")
    print()
    print(f"{Colors.GREEN}✓ Image Job:{Colors.END} {image_job_id}")
    print(f"  Duration: {image_status['duration_ms']}ms")
    print(f"  Output: {image_path}")
    print()
    print(f"{Colors.BOLD}{Colors.GREEN}Success!{Colors.END} Full workflow completed.")
    print(
        f"Total time: {llm_status['duration_ms'] + image_status['duration_ms']}ms"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
