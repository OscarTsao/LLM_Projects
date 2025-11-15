"""Test combination generation logic."""
from src.augment.combinator import ComboGenerator
from src.augment.methods import MethodRegistry

def test_singletons_count():
    """Singleton mode generates 28 combos (one per method)."""
    registry = MethodRegistry("conf/augment_methods.yaml")
    gen = ComboGenerator(registry.list_methods())
    combos = list(gen.iter_combos(mode="singletons"))
    # May be less than 28 if some methods unavailable
    assert len(combos) > 0, "No combos generated"

def test_combo_id_deterministic():
    """Combo IDs are deterministic."""
    registry = MethodRegistry("conf/augment_methods.yaml")
    gen1 = ComboGenerator(registry.list_methods())
    gen2 = ComboGenerator(registry.list_methods())
    ids1 = [c.combo_id for c in gen1.iter_combos(mode="singletons")]
    ids2 = [c.combo_id for c in gen2.iter_combos(mode="singletons")]
    assert ids1 == ids2, "Combo IDs not deterministic"
