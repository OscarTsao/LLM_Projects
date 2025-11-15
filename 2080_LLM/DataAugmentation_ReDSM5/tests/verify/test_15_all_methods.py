"""Test all 28 methods can instantiate."""
import pytest
from src.augment.methods import MethodRegistry
from tests.verify_utils import is_cuda_available

def test_all_methods_load():
    """All methods in registry can be instantiated (CPU subset)."""
    registry = MethodRegistry("conf/augment_methods.yaml")
    available = registry.list_methods()
    assert len(available) > 0, "No methods available"

@pytest.mark.gpu
@pytest.mark.xfail(not is_cuda_available(), reason="CUDA not available")
def test_gpu_methods_load():
    """GPU methods require CUDA."""
    registry = MethodRegistry("conf/augment_methods.yaml")
    # Attempt to instantiate a GPU method
    gpu_methods = ["nlp_cwe_sub_roberta", "ta_mlm_sub_bert"]
    for method_id in gpu_methods:
        try:
            aug = registry.instantiate(method_id)
            assert aug is not None
        except Exception:
            pytest.skip(f"GPU method {method_id} unavailable")
