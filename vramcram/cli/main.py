"""CLI tool for VramCram system control and job management."""

import asyncio
import json
import sys
from pathlib import Path

import click
import requests

from vramcram.main import main as vramcram_main


@click.group()
def cli() -> None:
    """VramCram GPU orchestration CLI."""
    pass


@cli.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    default=Path("config.yaml"),
    help="Path to configuration file",
)
def start(config: Path) -> None:
    """Start VramCram system."""
    click.echo(f"Starting VramCram with config: {config}")
    try:
        asyncio.run(vramcram_main(config))
    except KeyboardInterrupt:
        click.echo("\nShutdown requested")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("model")
@click.argument("prompt")
@click.option(
    "--host",
    default="localhost",
    help="API server host",
)
@click.option(
    "--port",
    default=8000,
    help="API server port",
)
@click.option(
    "--max-tokens",
    type=int,
    help="Maximum tokens to generate (LLM only)",
)
@click.option(
    "--temperature",
    type=float,
    help="Sampling temperature (LLM only)",
)
@click.option(
    "--width",
    type=int,
    help="Image width (diffusion only)",
)
@click.option(
    "--height",
    type=int,
    help="Image height (diffusion only)",
)
def submit(
    model: str,
    prompt: str,
    host: str,
    port: int,
    max_tokens: int | None,
    temperature: float | None,
    width: int | None,
    height: int | None,
) -> None:
    """Submit a job to VramCram.

    MODEL: Target model name
    PROMPT: Inference prompt/input
    """
    # Build params
    params = {}
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if temperature is not None:
        params["temperature"] = temperature
    if width is not None:
        params["width"] = width
    if height is not None:
        params["height"] = height

    # Submit job
    url = f"http://{host}:{port}/jobs"
    payload = {
        "model": model,
        "prompt": prompt,
        "params": params,
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()

        data = response.json()
        job_id = data["job_id"]

        click.echo(f"Job submitted successfully!")
        click.echo(f"Job ID: {job_id}")
        click.echo(f"Model: {data['model']}")
        click.echo(f"Status: {data['status']}")
        click.echo(f"\nCheck status with: vramcram-cli status {job_id}")

    except requests.exceptions.RequestException as e:
        click.echo(f"Error submitting job: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("job_id")
@click.option(
    "--host",
    default="localhost",
    help="API server host",
)
@click.option(
    "--port",
    default=8000,
    help="API server port",
)
@click.option(
    "--watch",
    is_flag=True,
    help="Watch job status until completion",
)
def status(job_id: str, host: str, port: int, watch: bool) -> None:
    """Check job status.

    JOB_ID: Job identifier
    """
    url = f"http://{host}:{port}/jobs/{job_id}"

    try:
        if watch:
            # Poll until completion
            import time

            click.echo("Watching job status (Ctrl+C to stop)...")

            while True:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()

                status_str = data["status"]
                click.echo(f"Status: {status_str}", nl=False)

                if status_str in ("completed", "failed"):
                    click.echo()
                    _print_job_status(data)
                    break

                click.echo("\r", nl=False)
                time.sleep(2)

        else:
            # Single status check
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            _print_job_status(data)

    except requests.exceptions.RequestException as e:
        click.echo(f"Error getting job status: {e}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nStopped watching")


def _print_job_status(data: dict) -> None:
    """Print job status in formatted way."""
    click.echo(f"Job ID: {data['job_id']}")
    click.echo(f"Model: {data['model']}")
    click.echo(f"Status: {data['status']}")
    click.echo(f"Created: {data['created_at']}")

    if data.get("started_at"):
        click.echo(f"Started: {data['started_at']}")

    if data.get("completed_at"):
        click.echo(f"Completed: {data['completed_at']}")

    if data.get("duration_ms"):
        click.echo(f"Duration: {data['duration_ms']}ms")

    if data.get("error"):
        click.echo(f"Error: {data['error']}", err=True)

    if data["status"] == "completed":
        click.echo(f"\nGet result with: vramcram-cli result {data['job_id']}")


@cli.command()
@click.argument("job_id")
@click.option(
    "--host",
    default="localhost",
    help="API server host",
)
@click.option(
    "--port",
    default=8000,
    help="API server port",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Save result to file",
)
def result(job_id: str, host: str, port: int, output: Path | None) -> None:
    """Get job result.

    JOB_ID: Job identifier
    """
    url = f"http://{host}:{port}/jobs/{job_id}/result"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data["status"] != "completed":
            click.echo(f"Job status: {data['status']}")
            if data.get("error"):
                click.echo(f"Error: {data['error']}", err=True)
            return

        result_data = data.get("result")

        if result_data is None:
            click.echo("No result available")
            return

        # Try to parse as JSON
        try:
            result_dict = json.loads(result_data)

            # Check if it's an image path
            if "image_path" in result_dict:
                click.echo(f"Image saved to: {result_dict['image_path']}")
            # Check if it's text
            elif "text" in result_dict:
                text = result_dict["text"]
                if output:
                    output.write_text(text)
                    click.echo(f"Result saved to: {output}")
                else:
                    click.echo("Result:")
                    click.echo(text)
            else:
                click.echo("Result:")
                click.echo(json.dumps(result_dict, indent=2))

        except json.JSONDecodeError:
            # Not JSON, print as-is
            if output:
                output.write_text(result_data)
                click.echo(f"Result saved to: {output}")
            else:
                click.echo("Result:")
                click.echo(result_data)

    except requests.exceptions.RequestException as e:
        click.echo(f"Error getting job result: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--host",
    default="localhost",
    help="API server host",
)
@click.option(
    "--port",
    default=8000,
    help="API server port",
)
def models(host: str, port: int) -> None:
    """List available models."""
    url = f"http://{host}:{port}/models"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data:
            click.echo("No models available")
            return

        click.echo(f"Available models ({len(data)}):\n")

        for model in data:
            loaded = "✓" if model["loaded"] else "○"
            click.echo(f"{loaded} {model['name']}")
            click.echo(f"  Type: {model['type']}")
            click.echo(f"  VRAM: {model['vram_mb']}MB")
            click.echo()

    except requests.exceptions.RequestException as e:
        click.echo(f"Error listing models: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--host",
    default="localhost",
    help="API server host",
)
@click.option(
    "--port",
    default=8000,
    help="API server port",
)
def health(host: str, port: int) -> None:
    """Check system health."""
    url = f"http://{host}:{port}/health"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        status_emoji = "✓" if data["status"] == "healthy" else "✗"
        click.echo(f"{status_emoji} System Status: {data['status']}")
        click.echo(f"  Redis: {'connected' if data['redis_connected'] else 'disconnected'}")
        click.echo(f"  Models: {data['models_available']}")
        click.echo(f"  Jobs Queued: {data['jobs_queued']}")

    except requests.exceptions.RequestException as e:
        click.echo(f"✗ System Status: unhealthy", err=True)
        click.echo(f"  Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
