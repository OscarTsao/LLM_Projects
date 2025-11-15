#!/usr/bin/env python
"""Test script for Phase 19: Security & Authentication.

This script tests:
1. API key creation and validation
2. Key expiration and revocation
3. Rate limiting
4. Authentication middleware
5. Security headers
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from psy_agents_noaug.security import (
    APIKeyAuth,
    AuthManager,
    RateLimitConfig,
    RateLimiter,
    add_security_headers,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


def test_api_key_creation() -> bool:
    """Test API key creation.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 1: API Key Creation")
    LOGGER.info("=" * 80)

    try:
        auth = APIKeyAuth()

        # Create key
        api_key = auth.create_key(
            name="test-key",
            expires_days=30,
            scopes=["read", "write"],
        )

        assert api_key.startswith("psy_")
        assert len(api_key) > 20

        # List keys
        keys = auth.list_keys()
        assert len(keys) == 1
        assert keys[0]["name"] == "test-key"
        assert keys[0]["scopes"] == ["read", "write"]

        LOGGER.info("✅ API Key Creation: PASSED")
        LOGGER.info(f"   - Created key: {api_key[:20]}...")
        LOGGER.info(f"   - Scopes: {keys[0]['scopes']}")

    except Exception:
        LOGGER.exception("❌ API Key Creation: FAILED")
        return False
    else:
        return True


def test_api_key_validation() -> bool:
    """Test API key validation.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 2: API Key Validation")
    LOGGER.info("=" * 80)

    try:
        auth = APIKeyAuth()

        # Create valid key
        api_key = auth.create_key(name="valid-key")

        # Test valid key
        is_valid, metadata = auth.verify_key(api_key)
        assert is_valid is True
        assert metadata is not None
        assert metadata["name"] == "valid-key"
        assert metadata["use_count"] == 1

        # Test invalid key
        is_valid, metadata = auth.verify_key("psy_invalid_key")
        assert is_valid is False
        assert metadata is None

        LOGGER.info("✅ API Key Validation: PASSED")
        LOGGER.info("   - Valid key accepted")
        LOGGER.info("   - Invalid key rejected")

    except Exception:
        LOGGER.exception("❌ API Key Validation: FAILED")
        return False
    else:
        return True


def test_api_key_expiration() -> bool:
    """Test API key expiration.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 3: API Key Expiration")
    LOGGER.info("=" * 80)

    try:
        auth = APIKeyAuth()

        # Create key that expires immediately
        api_key = auth.create_key(name="expired-key", expires_days=0)

        # Manually set expiration to past
        key_hash = auth._hash_key(api_key)
        auth.keys[key_hash]["expires_at"] = datetime.now() - timedelta(days=1)

        # Test expired key
        is_valid, _ = auth.verify_key(api_key)
        assert is_valid is False

        LOGGER.info("✅ API Key Expiration: PASSED")
        LOGGER.info("   - Expired key correctly rejected")

    except Exception:
        LOGGER.exception("❌ API Key Expiration: FAILED")
        return False
    else:
        return True


def test_api_key_revocation() -> bool:
    """Test API key revocation.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 4: API Key Revocation")
    LOGGER.info("=" * 80)

    try:
        auth = APIKeyAuth()

        # Create and revoke key
        api_key = auth.create_key(name="revoked-key")

        # Verify before revocation
        is_valid, _ = auth.verify_key(api_key)
        assert is_valid is True

        # Revoke
        revoked = auth.revoke_key(api_key)
        assert revoked is True

        # Verify after revocation
        is_valid, _ = auth.verify_key(api_key)
        assert is_valid is False

        LOGGER.info("✅ API Key Revocation: PASSED")
        LOGGER.info("   - Key successfully revoked")
        LOGGER.info("   - Revoked key rejected")

    except Exception:
        LOGGER.exception("❌ API Key Revocation: FAILED")
        return False
    else:
        return True


def test_rate_limiting() -> bool:
    """Test rate limiting.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 5: Rate Limiting")
    LOGGER.info("=" * 80)

    try:
        # Create limiter with low limits for testing
        limiter = RateLimiter(
            default_config=RateLimitConfig(
                requests=5,
                window_seconds=10,
            )
        )

        # Make requests within limit
        for _i in range(5):
            allowed, info = limiter.check_limit("test-client")
            assert allowed is True
            assert info["remaining"] >= 0

        # Next request should be rate limited
        allowed, info = limiter.check_limit("test-client")
        assert allowed is False
        assert info["retry_after"] is not None

        LOGGER.info("✅ Rate Limiting: PASSED")
        LOGGER.info("   - Requests within limit allowed")
        LOGGER.info(
            f"   - Rate limit enforced (retry after {info['retry_after']:.1f}s)"
        )

    except Exception:
        LOGGER.exception("❌ Rate Limiting: FAILED")
        return False
    else:
        return True


def test_rate_limit_recovery() -> bool:
    """Test rate limit recovery after waiting.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 6: Rate Limit Recovery")
    LOGGER.info("=" * 80)

    try:
        limiter = RateLimiter(
            default_config=RateLimitConfig(
                requests=2,
                window_seconds=1,
            )
        )

        # Exhaust limit
        limiter.check_limit("recovery-client")
        limiter.check_limit("recovery-client")

        # Should be limited
        allowed, _ = limiter.check_limit("recovery-client")
        assert allowed is False

        # Wait for recovery
        time.sleep(1.5)

        # Should be allowed again
        allowed, info = limiter.check_limit("recovery-client")
        assert allowed is True

        LOGGER.info("✅ Rate Limit Recovery: PASSED")
        LOGGER.info("   - Tokens successfully refilled")
        LOGGER.info(
            f"   - Requests allowed after recovery ({info['remaining']} remaining)"
        )

    except Exception:
        LOGGER.exception("❌ Rate Limit Recovery: FAILED")
        return False
    else:
        return True


def test_endpoint_specific_limits() -> bool:
    """Test endpoint-specific rate limits.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 7: Endpoint-Specific Limits")
    LOGGER.info("=" * 80)

    try:
        limiter = RateLimiter(
            default_config=RateLimitConfig(requests=100, window_seconds=60)
        )

        # Set strict limit for expensive endpoint
        limiter.set_endpoint_limit(
            "/predict/batch",
            RateLimitConfig(requests=2, window_seconds=60),
        )

        # Regular endpoint should have default limit
        for _i in range(10):
            allowed, _ = limiter.check_limit("client1", endpoint="/predict")
            assert allowed is True

        # Batch endpoint should have strict limit (2 requests per 60s)
        # With burst multiplier 1.5, we get 3 initial tokens
        allowed, _ = limiter.check_limit("client2", endpoint="/predict/batch")
        assert allowed is True
        allowed, _ = limiter.check_limit("client2", endpoint="/predict/batch")
        assert allowed is True
        allowed, _ = limiter.check_limit("client2", endpoint="/predict/batch")
        assert allowed is True  # Burst allows 3rd request
        allowed, _ = limiter.check_limit("client2", endpoint="/predict/batch")
        assert allowed is False  # 4th request is limited

        LOGGER.info("✅ Endpoint-Specific Limits: PASSED")
        LOGGER.info("   - Default limits applied correctly")
        LOGGER.info("   - Strict limits enforced for batch endpoint")

    except Exception:
        LOGGER.exception("❌ Endpoint-Specific Limits: FAILED")
        return False
    else:
        return True


def test_auth_manager() -> bool:
    """Test authentication manager.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 8: Authentication Manager")
    LOGGER.info("=" * 80)

    try:
        manager = AuthManager()

        # Create API key
        api_key = manager.api_key_auth.create_key(
            name="manager-test",
            scopes=["read", "write"],
        )

        # Test authentication with valid key
        is_auth, context = manager.authenticate(api_key=api_key)
        assert is_auth is True
        assert context is not None
        assert context["user"] == "manager-test"

        # Test authentication with invalid key
        is_auth, context = manager.authenticate(api_key="invalid")
        assert is_auth is False
        assert context is None

        # Test scope requirements
        is_auth, context = manager.authenticate(
            api_key=api_key,
            required_scopes=["read"],
        )
        assert is_auth is True

        is_auth, context = manager.authenticate(
            api_key=api_key,
            required_scopes=["admin"],
        )
        assert is_auth is False

        LOGGER.info("✅ Authentication Manager: PASSED")
        LOGGER.info("   - Valid authentication successful")
        LOGGER.info("   - Invalid authentication rejected")
        LOGGER.info("   - Scope requirements enforced")

    except Exception:
        LOGGER.exception("❌ Authentication Manager: FAILED")
        return False
    else:
        return True


def test_security_headers() -> bool:
    """Test security headers.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 9: Security Headers")
    LOGGER.info("=" * 80)

    try:
        # Test header addition
        headers: list[tuple[bytes, bytes]] = []
        updated_headers = add_security_headers(headers, hsts=True, csp=True)

        # Check for required security headers
        header_dict = {name.decode(): value.decode() for name, value in updated_headers}

        assert "X-Content-Type-Options" in header_dict
        assert header_dict["X-Content-Type-Options"] == "nosniff"

        assert "X-Frame-Options" in header_dict
        assert header_dict["X-Frame-Options"] == "DENY"

        assert "Strict-Transport-Security" in header_dict
        assert "Content-Security-Policy" in header_dict

        LOGGER.info("✅ Security Headers: PASSED")
        LOGGER.info(f"   - Added {len(updated_headers)} security headers")
        LOGGER.info("   - HSTS enabled")
        LOGGER.info("   - CSP enabled")

    except Exception:
        LOGGER.exception("❌ Security Headers: FAILED")
        return False
    else:
        return True


def main():
    """Run all security tests."""
    LOGGER.info("Starting Phase 19 Security Tests")
    LOGGER.info("=" * 80)

    # Run tests
    tests = [
        ("API Key Creation", test_api_key_creation),
        ("API Key Validation", test_api_key_validation),
        ("API Key Expiration", test_api_key_expiration),
        ("API Key Revocation", test_api_key_revocation),
        ("Rate Limiting", test_rate_limiting),
        ("Rate Limit Recovery", test_rate_limit_recovery),
        ("Endpoint-Specific Limits", test_endpoint_specific_limits),
        ("Authentication Manager", test_auth_manager),
        ("Security Headers", test_security_headers),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception:
            LOGGER.exception(f"Test '{test_name}' crashed")
            results.append((test_name, False))

    # Summary
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("TEST SUMMARY")
    LOGGER.info("=" * 80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        LOGGER.info(f"{status}: {test_name}")

    LOGGER.info("=" * 80)
    LOGGER.info(f"Results: {passed}/{total} tests passed")

    if passed == total:
        LOGGER.info("🎉 All tests passed!")
        return 0

    LOGGER.error(f"❌ {total - passed} test(s) failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
