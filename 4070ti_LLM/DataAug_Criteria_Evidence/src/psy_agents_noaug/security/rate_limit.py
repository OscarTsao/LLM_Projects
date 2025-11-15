#!/usr/bin/env python
"""Rate limiting and throttling (Phase 19).

This module provides:
- Token bucket rate limiting
- Per-user and global rate limits
- Configurable time windows
- Rate limit status tracking
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger(__name__)


class RateLimitExceededError(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: float):
        """Initialize exception.

        Args:
            message: Error message
            retry_after: Seconds until retry allowed
        """
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    requests: int  # Number of requests allowed
    window_seconds: int  # Time window in seconds
    burst_multiplier: float = 1.5  # Allow burst up to this multiple


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(
        self,
        default_config: RateLimitConfig | None = None,
    ):
        """Initialize rate limiter.

        Args:
            default_config: Default rate limit configuration
        """
        self.default_config = default_config or RateLimitConfig(
            requests=100,
            window_seconds=60,
        )

        # Track requests per client
        self.clients: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "tokens": self.default_config.requests,
                "last_update": time.time(),
                "total_requests": 0,
                "rejected_requests": 0,
            }
        )

        # Per-endpoint configurations
        self.endpoint_configs: dict[str, RateLimitConfig] = {}

        LOGGER.info("Initialized RateLimiter")

    def set_endpoint_limit(self, endpoint: str, config: RateLimitConfig) -> None:
        """Set rate limit for specific endpoint.

        Args:
            endpoint: Endpoint path
            config: Rate limit configuration
        """
        self.endpoint_configs[endpoint] = config
        LOGGER.info(
            f"Set rate limit for {endpoint}: "
            f"{config.requests} requests per {config.window_seconds}s"
        )

    def check_limit(
        self,
        client_id: str,
        endpoint: str | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Check if request is within rate limit.

        Args:
            client_id: Unique client identifier
            endpoint: Optional endpoint for specific limits

        Returns:
            Tuple of (is_allowed, limit_info)
        """
        # Get config for endpoint or use default
        config = (
            self.endpoint_configs.get(endpoint) if endpoint else None
        ) or self.default_config

        # Get client state
        client = self.clients[client_id]

        # Calculate token refill
        now = time.time()
        elapsed = now - client["last_update"]
        refill_rate = config.requests / config.window_seconds
        tokens_to_add = elapsed * refill_rate

        # Update tokens (cap at burst limit)
        max_tokens = config.requests * config.burst_multiplier
        client["tokens"] = min(
            max_tokens,
            client["tokens"] + tokens_to_add,
        )
        client["last_update"] = now

        # Check if request allowed
        if client["tokens"] >= 1.0:
            # Allow request
            client["tokens"] -= 1.0
            client["total_requests"] += 1

            limit_info = {
                "allowed": True,
                "remaining": int(client["tokens"]),
                "limit": config.requests,
                "window_seconds": config.window_seconds,
                "retry_after": None,
            }

            return True, limit_info

        # Rate limit exceeded
        client["rejected_requests"] += 1

        # Calculate retry after
        tokens_needed = 1.0 - client["tokens"]
        retry_after = tokens_needed / refill_rate

        limit_info = {
            "allowed": False,
            "remaining": 0,
            "limit": config.requests,
            "window_seconds": config.window_seconds,
            "retry_after": retry_after,
        }

        LOGGER.warning(
            f"Rate limit exceeded for client {client_id}: "
            f"retry after {retry_after:.1f}s"
        )

        return False, limit_info

    def consume(
        self,
        client_id: str,
        endpoint: str | None = None,
        cost: int = 1,
    ) -> None:
        """Consume rate limit tokens or raise exception.

        Args:
            client_id: Unique client identifier
            endpoint: Optional endpoint for specific limits
            cost: Number of tokens to consume

        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        allowed, info = self.check_limit(client_id, endpoint)

        if not allowed:
            raise RateLimitExceededError(
                f"Rate limit exceeded. Retry after {info['retry_after']:.1f}s",
                retry_after=info["retry_after"],
            )

        # Consume additional tokens if cost > 1
        if cost > 1:
            client = self.clients[client_id]
            client["tokens"] -= cost - 1

    def get_client_stats(self, client_id: str) -> dict[str, Any]:
        """Get statistics for a client.

        Args:
            client_id: Client identifier

        Returns:
            Client statistics
        """
        if client_id not in self.clients:
            return {
                "total_requests": 0,
                "rejected_requests": 0,
                "current_tokens": self.default_config.requests,
            }

        client = self.clients[client_id]

        return {
            "total_requests": client["total_requests"],
            "rejected_requests": client["rejected_requests"],
            "current_tokens": int(client["tokens"]),
            "rejection_rate": (
                client["rejected_requests"] / client["total_requests"]
                if client["total_requests"] > 0
                else 0.0
            ),
        }

    def reset_client(self, client_id: str) -> None:
        """Reset rate limit for a client.

        Args:
            client_id: Client identifier
        """
        if client_id in self.clients:
            del self.clients[client_id]
            LOGGER.info(f"Reset rate limit for client {client_id}")

    def get_summary(self) -> dict[str, Any]:
        """Get rate limiter summary.

        Returns:
            Summary statistics
        """
        total_requests = sum(c["total_requests"] for c in self.clients.values())
        total_rejected = sum(c["rejected_requests"] for c in self.clients.values())

        return {
            "total_clients": len(self.clients),
            "total_requests": total_requests,
            "total_rejected": total_rejected,
            "rejection_rate": (
                total_rejected / total_requests if total_requests > 0 else 0.0
            ),
            "default_limit": {
                "requests": self.default_config.requests,
                "window_seconds": self.default_config.window_seconds,
            },
            "endpoint_limits": {
                endpoint: {
                    "requests": config.requests,
                    "window_seconds": config.window_seconds,
                }
                for endpoint, config in self.endpoint_configs.items()
            },
        }


# Convenience function
def check_rate_limit(
    client_id: str,
    requests: int = 100,
    window_seconds: int = 60,
) -> bool:
    """Check rate limit (convenience function).

    Args:
        client_id: Client identifier
        requests: Requests allowed per window
        window_seconds: Time window in seconds

    Returns:
        True if within limit
    """
    limiter = RateLimiter(
        default_config=RateLimitConfig(
            requests=requests,
            window_seconds=window_seconds,
        )
    )

    allowed, _ = limiter.check_limit(client_id)
    return allowed
