#!/usr/bin/env python3
"""Demo script for VramCram API endpoints.

Quick demonstration of all API endpoints with example requests.

Usage:
    python scripts/demo_api.py --host localhost --port 8000
"""

import argparse
import json
import sys

import requests


def demo_health(base_url: str) -> None:
    """Demo health endpoint."""
    print("\n" + "=" * 60)
    print("GET /health")
    print("=" * 60)

    response = requests.get(f"{base_url}/health")
    print(f"Status: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")


def demo_models(base_url: str) -> None:
    """Demo models endpoint."""
    print("\n" + "=" * 60)
    print("GET /models")
    print("=" * 60)

    response = requests.get(f"{base_url}/models")
    print(f"Status: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")


def demo_submit_job(base_url: str, model: str) -> str | None:
    """Demo job submission.

    Returns:
        Job ID if successful, None otherwise.
    """
    print("\n" + "=" * 60)
    print("POST /jobs")
    print("=" * 60)

    payload = {
        "model": model,
        "prompt": "Write a haiku about GPU orchestration",
        "params": {"max_tokens": 50, "temperature": 0.7},
    }

    print(f"Request:\n{json.dumps(payload, indent=2)}")

    response = requests.post(f"{base_url}/jobs", json=payload)
    print(f"\nStatus: {response.status_code}")
    data = response.json()
    print(f"Response:\n{json.dumps(data, indent=2)}")

    return data.get("job_id")


def demo_job_status(base_url: str, job_id: str) -> None:
    """Demo job status endpoint."""
    print("\n" + "=" * 60)
    print(f"GET /jobs/{job_id}")
    print("=" * 60)

    response = requests.get(f"{base_url}/jobs/{job_id}")
    print(f"Status: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")


def demo_job_result(base_url: str, job_id: str) -> None:
    """Demo job result endpoint."""
    print("\n" + "=" * 60)
    print(f"GET /jobs/{job_id}/result")
    print("=" * 60)

    response = requests.get(f"{base_url}/jobs/{job_id}/result")
    print(f"Status: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")


def main() -> int:
    """Run API demo."""
    parser = argparse.ArgumentParser(description="Demo VramCram API endpoints")
    parser.add_argument("--host", default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument(
        "--model",
        help="Model name for job submission (auto-detected if not specified)",
    )

    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    print("VramCram API Demonstration")
    print("=" * 60)

    try:
        # 1. Health check
        demo_health(base_url)

        # 2. List models
        demo_models(base_url)

        # Get first model if not specified
        if not args.model:
            response = requests.get(f"{base_url}/models")
            models = response.json()
            if models:
                args.model = models[0]["name"]
                print(f"\nUsing first available model: {args.model}")
            else:
                print("\nNo models available for job submission demo")
                return 0

        # 3. Submit job
        job_id = demo_submit_job(base_url, args.model)

        if job_id:
            # 4. Check status
            demo_job_status(base_url, job_id)

            # 5. Get result (will show "not completed" unless job finishes quickly)
            demo_job_result(base_url, job_id)

            print("\n" + "=" * 60)
            print(f"Job ID for tracking: {job_id}")
            print("=" * 60)
            print("\nTo check job status later:")
            print(f"  curl {base_url}/jobs/{job_id}")
            print("\nTo get result when completed:")
            print(f"  curl {base_url}/jobs/{job_id}/result")

    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to VramCram API")
        print(f"Make sure VramCram is running at {base_url}")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
