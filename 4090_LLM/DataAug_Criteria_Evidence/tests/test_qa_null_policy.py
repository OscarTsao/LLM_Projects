def decide_null(policy, threshold=None, ratio=None, null_score=0.0, nonnull_score=1.0):
    if policy == "none":
        return False
    if policy == "threshold":
        return null_score >= (threshold or 0.0)
    if policy == "ratio":
        return (null_score + 1e-9) / (nonnull_score + 1e-9) >= (ratio or 0.5)
    if policy == "calibrated":
        # Placeholder for calibration logic; behaves like threshold here
        return null_score >= (threshold or 0.0)
    raise ValueError(policy)


def test_policies():
    assert decide_null("none") is False
    assert (
        decide_null("threshold", threshold=0.2, null_score=0.3, nonnull_score=0.1)
        is True
    )
    assert decide_null("ratio", ratio=0.5, null_score=0.6, nonnull_score=0.9) is True
    assert decide_null("calibrated", threshold=0.1, null_score=0.2) is True
