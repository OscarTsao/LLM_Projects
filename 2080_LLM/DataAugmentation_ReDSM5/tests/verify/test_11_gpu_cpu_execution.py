"""Test GPU/CPU execution policies."""
import pytest
from tests.verify_utils import is_cuda_available

@pytest.mark.gpu
@pytest.mark.xfail(not is_cuda_available(), reason="CUDA not available")
def test_cuda_available():
    """GPU tests require CUDA."""
    assert is_cuda_available()
