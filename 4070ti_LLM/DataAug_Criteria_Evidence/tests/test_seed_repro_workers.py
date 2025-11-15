"""Tests for worker seeding utilities."""

from psy_agents_noaug.augmentation.pipeline import (
    AugConfig,
    AugmenterPipeline,
    worker_init,
)


def test_worker_init_unique_seeds():
    base_seed = 123
    seeds = [worker_init(i, base_seed, num_workers_per_rank=4) for i in range(4)]
    assert len(set(seeds)) == 4


def test_pipeline_seed_reproducibility():
    cfg = AugConfig(
        lib="nlpaug",
        methods=["nlpaug/char/KeyboardAug"],
        p_apply=0.0,  # skip augmentation, focus on RNG determinism
        seed=42,
    )
    pipeline = AugmenterPipeline(cfg)

    pipeline.set_seed(101)
    sequence_a = [pipeline._rng.random() for _ in range(5)]
    pipeline.set_seed(101)
    sequence_b = [pipeline._rng.random() for _ in range(5)]

    assert sequence_a == sequence_b
