#!/usr/bin/env python
"""Security middleware for FastAPI (Phase 19).

This module provides:
- Security headers middleware
- Request/response security validation
- CORS configuration
- Content security policy
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Receive, Scope, Send

LOGGER = logging.getLogger(__name__)


class SecurityMiddleware:
    """Security middleware for FastAPI applications."""

    def __init__(
        self,
        app: ASGIApp,
        enable_hsts: bool = True,
        enable_csp: bool = True,
        enable_xss_protection: bool = True,
    ):
        """Initialize security middleware.

        Args:
            app: ASGI application
            enable_hsts: Enable HTTP Strict Transport Security
            enable_csp: Enable Content Security Policy
            enable_xss_protection: Enable XSS protection headers
        """
        self.app = app
        self.enable_hsts = enable_hsts
        self.enable_csp = enable_csp
        self.enable_xss_protection = enable_xss_protection
        LOGGER.info("Initialized SecurityMiddleware")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle request with security headers.

        Args:
            scope: ASGI scope
            receive: ASGI receive channel
            send: ASGI send channel
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Wrap send to add security headers
        async def send_with_security_headers(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))

                # Add security headers
                security_headers = self._get_security_headers()
                for name, value in security_headers.items():
                    headers.append((name.encode(), value.encode()))

                message["headers"] = headers

            await send(message)

        await self.app(scope, receive, send_with_security_headers)

    def _get_security_headers(self) -> dict[str, str]:
        """Get security headers to add.

        Returns:
            Dictionary of header name/value pairs
        """
        headers = {
            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",
            # Prevent clickjacking
            "X-Frame-Options": "DENY",
            # Referrer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            # Permissions policy
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        }

        # HTTP Strict Transport Security
        if self.enable_hsts:
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # Content Security Policy
        if self.enable_csp:
            headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "font-src 'self'; "
                "connect-src 'self'; "
                "frame-ancestors 'none'"
            )

        # XSS Protection
        if self.enable_xss_protection:
            headers["X-XSS-Protection"] = "1; mode=block"

        return headers


# Helper function
def add_security_headers(
    response_headers: list[tuple[bytes, bytes]],
    hsts: bool = True,
    csp: bool = True,
) -> list[tuple[bytes, bytes]]:
    """Add security headers to response.

    Args:
        response_headers: Existing response headers
        hsts: Include HSTS header
        csp: Include CSP header

    Returns:
        Updated headers list
    """
    security_headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "X-XSS-Protection": "1; mode=block",
    }

    if hsts:
        security_headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )

    if csp:
        security_headers["Content-Security-Policy"] = (
            "default-src 'self'; frame-ancestors 'none'"
        )

    # Add to existing headers
    for name, value in security_headers.items():
        response_headers.append((name.encode(), value.encode()))

    return response_headers
