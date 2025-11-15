#!/usr/bin/env python
"""Test script for Phase 18: API Serving & REST Endpoints.

This script tests:
1. API server startup and shutdown
2. Health and readiness endpoints
3. Metrics endpoint
4. Single prediction endpoint
5. Batch prediction endpoint
6. Error handling
"""

from __future__ import annotations

import logging
import multiprocessing
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Test configuration
API_HOST = "127.0.0.1"
API_PORT = 8888
BASE_URL = f"http://{API_HOST}:{API_PORT}"


def run_server_process():
    """Run API server in subprocess."""
    from psy_agents_noaug.api.server import run_server

    # Suppress uvicorn logs
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    run_server(host=API_HOST, port=API_PORT, reload=False)


def wait_for_server(timeout: int = 10) -> bool:
    """Wait for server to be ready.

    Args:
        timeout: Maximum wait time in seconds

    Returns:
        True if server is ready
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=1)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass

        time.sleep(0.5)

    return False


def test_root_endpoint() -> bool:
    """Test root endpoint.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 1: Root Endpoint")
    LOGGER.info("=" * 80)

    try:
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "status" in data

        LOGGER.info("✅ Root Endpoint: PASSED")
        LOGGER.info(f"   - Name: {data['name']}")
        LOGGER.info(f"   - Version: {data['version']}")
        LOGGER.info(f"   - Status: {data['status']}")

    except Exception:
        LOGGER.exception("❌ Root Endpoint: FAILED")
        return False
    else:
        return True


def test_health_endpoint() -> bool:
    """Test health check endpoint.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 2: Health Endpoint")
    LOGGER.info("=" * 80)

    try:
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "checks" in data
        assert "uptime_seconds" in data

        # Verify status is valid
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

        LOGGER.info("✅ Health Endpoint: PASSED")
        LOGGER.info(f"   - Status: {data['status']}")
        LOGGER.info(f"   - Checks: {len(data['checks'])}")
        LOGGER.info(f"   - Uptime: {data['uptime_seconds']:.2f}s")

    except Exception:
        LOGGER.exception("❌ Health Endpoint: FAILED")
        return False
    else:
        return True


def test_readiness_endpoint() -> bool:
    """Test readiness check endpoint.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 3: Readiness Endpoint")
    LOGGER.info("=" * 80)

    try:
        response = requests.get(f"{BASE_URL}/ready")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "ready"

        LOGGER.info("✅ Readiness Endpoint: PASSED")
        LOGGER.info(f"   - Status: {data['status']}")

    except Exception:
        LOGGER.exception("❌ Readiness Endpoint: FAILED")
        return False
    else:
        return True


def test_metrics_endpoint() -> bool:
    """Test metrics endpoint.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 4: Metrics Endpoint")
    LOGGER.info("=" * 80)

    try:
        response = requests.get(f"{BASE_URL}/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "timestamp" in data
        assert "performance" in data
        assert "requests" in data

        # Verify performance metrics
        perf = data["performance"]
        assert "latency_p50_ms" in perf
        assert "latency_p95_ms" in perf
        assert "throughput_rps" in perf

        LOGGER.info("✅ Metrics Endpoint: PASSED")
        LOGGER.info(f"   - Latency P50: {perf.get('latency_p50_ms', 0):.2f}ms")
        LOGGER.info(f"   - Latency P95: {perf.get('latency_p95_ms', 0):.2f}ms")
        LOGGER.info(f"   - Throughput: {perf.get('throughput_rps', 0):.2f} req/s")

    except Exception:
        LOGGER.exception("❌ Metrics Endpoint: FAILED")
        return False
    else:
        return True


def test_predict_endpoint() -> bool:
    """Test single prediction endpoint.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 5: Predict Endpoint")
    LOGGER.info("=" * 80)

    try:
        # Valid request
        payload = {
            "text": "Patient shows symptoms of depression.",
            "task": "criteria",
        }

        response = requests.post(f"{BASE_URL}/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "task" in data
        assert "model_version" in data
        assert "latency_ms" in data
        assert "timestamp" in data

        # Verify prediction structure
        assert data["task"] == "criteria"
        assert 0 <= data["confidence"] <= 1
        assert data["latency_ms"] > 0

        LOGGER.info("✅ Predict Endpoint: PASSED")
        LOGGER.info(f"   - Prediction: {data['prediction']}")
        LOGGER.info(f"   - Confidence: {data['confidence']:.4f}")
        LOGGER.info(f"   - Latency: {data['latency_ms']:.2f}ms")

    except Exception:
        LOGGER.exception("❌ Predict Endpoint: FAILED")
        return False
    else:
        return True


def test_predict_invalid_task() -> bool:
    """Test prediction with invalid task.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 6: Predict with Invalid Task")
    LOGGER.info("=" * 80)

    try:
        payload = {
            "text": "Test text",
            "task": "invalid_task",
        }

        response = requests.post(f"{BASE_URL}/predict", json=payload)
        assert response.status_code == 400  # Bad request

        data = response.json()
        assert "detail" in data

        LOGGER.info("✅ Predict with Invalid Task: PASSED")
        LOGGER.info(f"   - Error handled correctly: {data['detail']}")

    except Exception:
        LOGGER.exception("❌ Predict with Invalid Task: FAILED")
        return False
    else:
        return True


def test_batch_predict_endpoint() -> bool:
    """Test batch prediction endpoint.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 7: Batch Predict Endpoint")
    LOGGER.info("=" * 80)

    try:
        payload = {
            "texts": [
                "Patient shows symptoms of depression.",
                "No significant symptoms observed.",
                "Moderate anxiety reported.",
            ],
            "task": "criteria",
        }

        response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "predictions" in data
        assert "count" in data
        assert "total_latency_ms" in data
        assert "avg_latency_ms" in data
        assert "task" in data

        # Verify batch results
        assert data["count"] == len(payload["texts"])
        assert len(data["predictions"]) == data["count"]
        assert data["total_latency_ms"] > 0
        assert data["avg_latency_ms"] > 0

        LOGGER.info("✅ Batch Predict Endpoint: PASSED")
        LOGGER.info(f"   - Count: {data['count']}")
        LOGGER.info(f"   - Total latency: {data['total_latency_ms']:.2f}ms")
        LOGGER.info(f"   - Avg latency: {data['avg_latency_ms']:.2f}ms")

    except Exception:
        LOGGER.exception("❌ Batch Predict Endpoint: FAILED")
        return False
    else:
        return True


def test_request_validation() -> bool:
    """Test request validation.

    Returns:
        True if successful
    """
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 8: Request Validation")
    LOGGER.info("=" * 80)

    try:
        # Empty text
        payload = {"text": "", "task": "criteria"}
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        assert response.status_code == 422  # Validation error

        # Missing required field
        payload = {"task": "criteria"}
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        assert response.status_code == 422

        LOGGER.info("✅ Request Validation: PASSED")
        LOGGER.info("   - Empty text rejected correctly")
        LOGGER.info("   - Missing field rejected correctly")

    except Exception:
        LOGGER.exception("❌ Request Validation: FAILED")
        return False
    else:
        return True


def main():
    """Run all API tests."""
    LOGGER.info("Starting Phase 18 API Tests")
    LOGGER.info("=" * 80)

    # Start server in subprocess
    server_process = multiprocessing.Process(target=run_server_process)
    server_process.start()

    try:
        # Wait for server to be ready
        LOGGER.info("Waiting for server to start...")
        if not wait_for_server(timeout=15):
            LOGGER.error("Server failed to start within timeout")
            return 1

        LOGGER.info("Server is ready!")
        LOGGER.info("")

        # Run tests
        tests = [
            ("Root Endpoint", test_root_endpoint),
            ("Health Endpoint", test_health_endpoint),
            ("Readiness Endpoint", test_readiness_endpoint),
            ("Metrics Endpoint", test_metrics_endpoint),
            ("Predict Endpoint", test_predict_endpoint),
            ("Predict with Invalid Task", test_predict_invalid_task),
            ("Batch Predict Endpoint", test_batch_predict_endpoint),
            ("Request Validation", test_request_validation),
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

    finally:
        # Cleanup
        LOGGER.info("")
        LOGGER.info("Stopping server...")
        server_process.terminate()
        server_process.join(timeout=5)
        if server_process.is_alive():
            server_process.kill()
        LOGGER.info("Server stopped")


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    multiprocessing.set_start_method("spawn", force=True)
    sys.exit(main())
