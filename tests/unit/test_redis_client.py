"""Tests for Redis client factory."""

import pytest

from vramcram.config.models import RedisConfig
from vramcram.redis.client import RedisClientFactory


def test_redis_client_factory_creation() -> None:
    """Test creating Redis client factory."""
    config = RedisConfig(host="localhost", port=6379, db=0)
    factory = RedisClientFactory(config)

    assert factory._config == config
    assert factory._pool is not None


def test_redis_client_factory_create_client() -> None:
    """Test creating Redis clients from factory."""
    config = RedisConfig(host="localhost", port=6379, db=15)
    factory = RedisClientFactory(config)

    client = factory.create_client()
    assert client is not None

    # Test basic operation
    client.set("test_key", "test_value")
    assert client.get("test_key") == "test_value"

    # Clean up
    client.delete("test_key")
    client.close()
    factory.close()


def test_redis_client_factory_connection_pooling() -> None:
    """Test that clients share the connection pool."""
    config = RedisConfig(host="localhost", port=6379, db=15, max_connections=10)
    factory = RedisClientFactory(config)

    client1 = factory.create_client()
    client2 = factory.create_client()

    # Both should use the same connection pool
    assert client1.connection_pool is client2.connection_pool

    client1.close()
    client2.close()
    factory.close()


def test_redis_client_factory_with_password() -> None:
    """Test factory with password configuration."""
    config = RedisConfig(host="localhost", port=6379, db=0, password="secret")
    factory = RedisClientFactory(config)

    # Just verify factory creation (actual auth would fail without Redis password)
    assert factory._pool is not None

    factory.close()


def test_redis_client_factory_close() -> None:
    """Test closing the factory disconnects pool."""
    config = RedisConfig(host="localhost", port=6379, db=15)
    factory = RedisClientFactory(config)

    client = factory.create_client()
    client.set("test_key", "test_value")

    # Close factory
    factory.close()

    # Pool should be disconnected
    # Attempting operations may fail or behave unexpectedly
    client.close()
