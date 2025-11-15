"""Test MethodRegistry loads 28 methods correctly."""
from pathlib import Path
import yaml

def test_registry_yaml_exists():
    yaml_path = Path("conf/augment_methods.yaml")
    assert yaml_path.exists(), "Method registry YAML missing"

def test_registry_has_28_methods():
    with open("conf/augment_methods.yaml") as f:
        data = yaml.safe_load(f)
    methods = data["methods"]
    assert len(methods) == 28, f"Expected 28 methods, found {len(methods)}"

def test_method_specs_complete():
    with open("conf/augment_methods.yaml") as f:
        data = yaml.safe_load(f)
    for method in data["methods"]:
        assert "id" in method, f"Missing 'id' in method {method}"
        assert "lib" in method, f"Missing 'lib' in method {method['id']}"
        assert method["lib"] in ["nlpaug", "textattack"], f"Invalid lib: {method['lib']}"
        assert "kind" in method
        assert "args" in method
        assert "requires_gpu" in method
        assert isinstance(method["requires_gpu"], bool)

def test_gpu_methods_count():
    with open("conf/augment_methods.yaml") as f:
        data = yaml.safe_load(f)
    gpu_methods = [m for m in data["methods"] if m["requires_gpu"]]
    assert len(gpu_methods) == 5, f"Expected 5 GPU methods, found {len(gpu_methods)}"
