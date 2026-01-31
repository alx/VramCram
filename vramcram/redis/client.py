"""Redis client factory and connection management."""

import redis
from redis.connection import ConnectionPool

from vramcram.config.models import RedisConfig


class RedisClientFactory:
    """Factory for creating Redis clients with connection pooling."""

    def __init__(self, config: RedisConfig) -> None:
        """Initialize Redis client factory.

        Args:
            config: Redis configuration.
        """
        self._config = config
        self._pool = ConnectionPool(
            host=config.host,
            port=config.port,
            db=config.db,
            password=config.password,
            max_connections=config.max_connections,
            socket_timeout=config.socket_timeout,
            socket_connect_timeout=config.socket_connect_timeout,
            decode_responses=True,  # Auto-decode bytes to strings
        )

    def create_client(self) -> redis.Redis:  # type: ignore[type-arg]
        """Create a new Redis client from the pool.

        Returns:
            Redis client instance.
        """
        return redis.Redis(connection_pool=self._pool)

    def close(self) -> None:
        """Close the connection pool and all connections."""
        self._pool.disconnect()
