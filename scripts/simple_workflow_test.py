#!/usr/bin/env python3
"""Simple workflow test - LLM to Image generation.

A minimal test script for automated testing and CI/CD.

Usage:
    python scripts/simple_workflow_test.py
"""

import json
import sys
import time

import requests


def test_workflow(base_url: str = "http://localhost:8005") -> bool:
    """Test the full LLM-to-Image workflow.

    Args:
        base_url: Base URL of VramCram API.

    Returns:
        True if successful, False otherwise.
    """
    print("Starting workflow test...")

    # 1. Health check
    print("\n1. Checking health...")
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        health = resp.json()
        if health["status"] != "healthy":
            print(f"ERROR: System unhealthy: {health}")
            return False
        print(f"✓ System healthy ({health['models_available']} models)")
    except Exception as e:
        print(f"ERROR: Health check failed: {e}")
        return False

    # 2. List models
    print("\n2. Listing models...")
    try:
        resp = requests.get(f"{base_url}/models", timeout=5)
        models = resp.json()

        llm_models = [m["name"] for m in models if m["type"] == "llm"]
        diffusion_models = [m["name"] for m in models if m["type"] == "diffusion"]

        if not llm_models or not diffusion_models:
            print(f"ERROR: Missing models. LLM: {llm_models}, Diffusion: {diffusion_models}")
            return False

        llm_model = llm_models[0]
        diffusion_model = diffusion_models[0]

        print(f"✓ Using LLM: {llm_model}")
        print(f"✓ Using Diffusion: {diffusion_model}")
    except Exception as e:
        print(f"ERROR: Failed to list models: {e}")
        return False

    # 3. Submit LLM job
    print("\n3. Submitting LLM job...")
    try:
        resp = requests.post(
            f"{base_url}/jobs",
            json={
                "model": llm_model,
                "prompt": "Describe a beautiful sunset over mountains in vivid detail. Keep it to 2 sentences.",
                "params": {"max_tokens": 100, "temperature": 0.7},
            },
            timeout=10,
        )
        resp.raise_for_status()
        llm_job = resp.json()
        llm_job_id = llm_job["job_id"]
        print(f"✓ LLM job submitted: {llm_job_id}")
    except Exception as e:
        print(f"ERROR: Failed to submit LLM job: {e}")
        return False

    # 4. Wait for LLM completion
    print("\n4. Waiting for LLM completion...")
    llm_result = None
    for i in range(60):  # 60 * 5 = 300 seconds timeout
        try:
            resp = requests.get(f"{base_url}/jobs/{llm_job_id}", timeout=5)
            status_data = resp.json()

            if status_data["status"] == "completed":
                # Get result
                resp = requests.get(f"{base_url}/jobs/{llm_job_id}/result", timeout=5)
                result_data = resp.json()
                llm_result = json.loads(result_data["result"])
                print(f"✓ LLM completed in {status_data['duration_ms']}ms")
                print(f"  Generated: {llm_result['text'][:80]}...")
                break
            elif status_data["status"] == "failed":
                print(f"ERROR: LLM job failed: {status_data.get('error')}")
                return False

            if i % 5 == 0:
                print(f"  Status: {status_data['status']} ({i*5}s elapsed)")

            time.sleep(5)
        except Exception as e:
            print(f"ERROR: Failed to check LLM status: {e}")
            return False
    else:
        print("ERROR: LLM job timeout")
        return False

    if not llm_result or "text" not in llm_result:
        print("ERROR: No LLM result text")
        return False

    description = llm_result["text"].strip()

    # 5. Submit Image job using LLM output
    print("\n5. Submitting image generation job...")
    try:
        resp = requests.post(
            f"{base_url}/jobs",
            json={
                "model": diffusion_model,
                "prompt": description,  # Use LLM output as prompt
                "params": {"width": 512, "height": 512, "sample_steps": 4},
            },
            timeout=10,
        )
        resp.raise_for_status()
        image_job = resp.json()
        image_job_id = image_job["job_id"]
        print(f"✓ Image job submitted: {image_job_id}")
    except Exception as e:
        print(f"ERROR: Failed to submit image job: {e}")
        return False

    # 6. Wait for image completion
    print("\n6. Waiting for image generation...")
    image_result = None
    for i in range(60):  # 60 * 5 = 300 seconds timeout
        try:
            resp = requests.get(f"{base_url}/jobs/{image_job_id}", timeout=5)
            status_data = resp.json()

            if status_data["status"] == "completed":
                # Get result
                resp = requests.get(f"{base_url}/jobs/{image_job_id}/result", timeout=5)
                result_data = resp.json()
                image_result = json.loads(result_data["result"])
                print(f"✓ Image completed in {status_data['duration_ms']}ms")
                print(f"  Path: {image_result['image_path']}")
                break
            elif status_data["status"] == "failed":
                print(f"ERROR: Image job failed: {status_data.get('error')}")
                return False

            if i % 5 == 0:
                print(f"  Status: {status_data['status']} ({i*5}s elapsed)")

            time.sleep(5)
        except Exception as e:
            print(f"ERROR: Failed to check image status: {e}")
            return False
    else:
        print("ERROR: Image job timeout")
        return False

    if not image_result or "image_path" not in image_result:
        print("ERROR: No image result path")
        return False

    # Success!
    print("\n" + "=" * 60)
    print("WORKFLOW TEST PASSED!")
    print("=" * 60)
    print(f"LLM Job:   {llm_job_id}")
    print(f"Image Job: {image_job_id}")
    print(f"Output:    {image_result['image_path']}")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_workflow()
    sys.exit(0 if success else 1)
