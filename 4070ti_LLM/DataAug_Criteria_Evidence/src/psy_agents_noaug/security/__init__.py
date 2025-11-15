#!/usr/bin/env python
"""Security and authentication module (Phase 19).

This module provides production-ready security features including:
- API key authentication
- Rate limiting and throttling
- Security headers
- Token management
- Request validation

Key Features:
- Multiple authentication strategies
- Configurable rate limits
- Token-based access control
- Security best practices
"""

from __future__ import annotations

from psy_agents_noaug.security.auth import (
    APIKeyAuth,
    AuthManager,
    create_api_key,
    verify_api_key,
)
from psy_agents_noaug.security.middleware import (
    SecurityMiddleware,
    add_security_headers,
)
from psy_agents_noaug.security.rate_limit import (
    RateLimitConfig,
    RateLimiter,
    RateLimitExceededError,
    check_rate_limit,
)

__all__ = [
    # Authentication
    "APIKeyAuth",
    "AuthManager",
    "create_api_key",
    "verify_api_key",
    # Rate Limiting
    "RateLimitConfig",
    "RateLimiter",
    "RateLimitExceeded",
    "check_rate_limit",
    # Middleware
    "SecurityMiddleware",
    "add_security_headers",
]
