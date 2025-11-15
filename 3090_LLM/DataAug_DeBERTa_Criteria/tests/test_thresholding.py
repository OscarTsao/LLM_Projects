import numpy as np

from src.dataaug_multi_both.training.metrics import optimize_global_threshold


def test_opt_global_threshold():
    y_true = np.array([[1,0,1],[0,1,0],[1,1,0],[0,0,1]])
    y_prob = np.array([[0.9,0.2,0.6],[0.1,0.8,0.4],[0.7,0.6,0.3],[0.2,0.3,0.9]])
    t, f1 = optimize_global_threshold(y_true, y_prob, 0.2, 0.6, 21)
    assert 0.2 <= t <= 0.6
    assert f1 >= 0.5
