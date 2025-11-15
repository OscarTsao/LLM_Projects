#!/usr/bin/env python
"""Authentication and API key management (Phase 19).

This module provides:
- API key generation and validation
- Token-based authentication
- Key rotation and expiration
- Authentication middleware
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timedelta
from typing import Any

LOGGER = logging.getLogger(__name__)


class APIKeyAuth:
    """API key authentication handler."""

    def __init__(
        self,
        key_length: int = 32,
        key_prefix: str = "psy",
    ):
        """Initialize API key authenticator.

        Args:
            key_length: Length of generated API keys
            key_prefix: Prefix for API keys
        """
        self.key_length = key_length
        self.key_prefix = key_prefix
        self.keys: dict[str, dict[str, Any]] = {}
        LOGGER.info("Initialized APIKeyAuth")

    def create_key(
        self,
        name: str,
        expires_days: int | None = None,
        scopes: list[str] | None = None,
    ) -> str:
        """Create a new API key.

        Args:
            name: Descriptive name for the key
            expires_days: Days until expiration (None = no expiration)
            scopes: List of allowed scopes/permissions

        Returns:
            Generated API key
        """
        # Generate random key
        random_part = secrets.token_urlsafe(self.key_length)
        api_key = f"{self.key_prefix}_{random_part}"

        # Hash the key for storage
        key_hash = self._hash_key(api_key)

        # Store key metadata
        expires_at = None
        if expires_days:
            expires_at = datetime.now() + timedelta(days=expires_days)

        self.keys[key_hash] = {
            "name": name,
            "created_at": datetime.now(),
            "expires_at": expires_at,
            "scopes": scopes or [],
            "last_used": None,
            "use_count": 0,
        }

        LOGGER.info(f"Created API key: {name}")
        return api_key

    def verify_key(self, api_key: str) -> tuple[bool, dict[str, Any] | None]:
        """Verify an API key.

        Args:
            api_key: API key to verify

        Returns:
            Tuple of (is_valid, key_metadata)
        """
        key_hash = self._hash_key(api_key)

        if key_hash not in self.keys:
            return False, None

        metadata = self.keys[key_hash]

        # Check expiration
        if metadata["expires_at"] and datetime.now() > metadata["expires_at"]:
            LOGGER.warning(f"Expired API key used: {metadata['name']}")
            return False, None

        # Update usage stats
        metadata["last_used"] = datetime.now()
        metadata["use_count"] += 1

        return True, metadata

    def revoke_key(self, api_key: str) -> bool:
        """Revoke an API key.

        Args:
            api_key: API key to revoke

        Returns:
            True if key was revoked
        """
        key_hash = self._hash_key(api_key)

        if key_hash in self.keys:
            name = self.keys[key_hash]["name"]
            del self.keys[key_hash]
            LOGGER.info(f"Revoked API key: {name}")
            return True

        return False

    def list_keys(self) -> list[dict[str, Any]]:
        """List all API keys (excluding actual keys).

        Returns:
            List of key metadata
        """
        return [
            {
                "name": metadata["name"],
                "created_at": metadata["created_at"].isoformat(),
                "expires_at": (
                    metadata["expires_at"].isoformat()
                    if metadata["expires_at"]
                    else None
                ),
                "scopes": metadata["scopes"],
                "last_used": (
                    metadata["last_used"].isoformat() if metadata["last_used"] else None
                ),
                "use_count": metadata["use_count"],
            }
            for metadata in self.keys.values()
        ]

    def _hash_key(self, api_key: str) -> str:
        """Hash an API key for secure storage.

        Args:
            api_key: API key to hash

        Returns:
            Hashed key
        """
        return hashlib.sha256(api_key.encode()).hexdigest()


class AuthManager:
    """Centralized authentication manager."""

    def __init__(self):
        """Initialize authentication manager."""
        self.api_key_auth = APIKeyAuth()
        self.allow_anonymous = False
        LOGGER.info("Initialized AuthManager")

    def authenticate(
        self,
        api_key: str | None = None,
        required_scopes: list[str] | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Authenticate a request.

        Args:
            api_key: API key from request
            required_scopes: Required permission scopes

        Returns:
            Tuple of (is_authenticated, auth_context)
        """
        # Allow anonymous access if configured
        if not api_key and self.allow_anonymous:
            return True, {"user": "anonymous", "scopes": []}

        # Require API key
        if not api_key:
            return False, None

        # Verify API key
        is_valid, metadata = self.api_key_auth.verify_key(api_key)

        if not is_valid:
            return False, None

        # Check scopes
        if required_scopes:
            key_scopes = set(metadata["scopes"])  # type: ignore[index]
            required = set(required_scopes)

            if not required.issubset(key_scopes):
                LOGGER.warning(
                    f"Insufficient scopes for {metadata['name']}: "  # type: ignore[index]
                    f"required {required}, has {key_scopes}"
                )
                return False, None

        # Build auth context
        auth_context = {
            "user": metadata["name"],  # type: ignore[index]
            "scopes": metadata["scopes"],  # type: ignore[index]
            "authenticated_at": datetime.now().isoformat(),
        }

        return True, auth_context


# Convenience functions
def create_api_key(
    name: str,
    expires_days: int | None = None,
    scopes: list[str] | None = None,
) -> str:
    """Create an API key (convenience function).

    Args:
        name: Descriptive name for the key
        expires_days: Days until expiration
        scopes: List of allowed scopes

    Returns:
        Generated API key
    """
    auth = APIKeyAuth()
    return auth.create_key(name=name, expires_days=expires_days, scopes=scopes)


def verify_api_key(api_key: str) -> bool:
    """Verify an API key (convenience function).

    Args:
        api_key: API key to verify

    Returns:
        True if valid
    """
    auth = APIKeyAuth()
    is_valid, _ = auth.verify_key(api_key)
    return is_valid
