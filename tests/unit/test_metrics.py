"""Tests for evaluation metrics."""
import numpy as np
import pytest

m = pytest.importorskip("src.evaluation.metrics")


def _rand(n=17, seed=0):
    return np.random.default_rng(seed).standard_normal((n, 3))


def test_mpjpe_zero_on_identical():
    gt = _rand()
    np.testing.assert_allclose(m.compute_mpjpe(gt, gt), 0.0, atol=1e-8)


def test_mpjpe_known_value():
    gt = _rand()
    pred = gt.copy()
    pred[:, 0] += 1.0
    np.testing.assert_allclose(m.compute_mpjpe(pred, gt), 1.0, atol=1e-5)


def test_pa_mpjpe_zero_on_identical():
    gt = _rand()
    np.testing.assert_allclose(m.compute_pa_mpjpe(gt, gt), 0.0, atol=1e-5)


def test_pa_mpjpe_rotation_invariant():
    gt = _rand()
    angle = np.radians(45)
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle),  np.cos(angle), 0],
                  [0, 0, 1]])
    pred = (R @ gt.T).T
    assert m.compute_pa_mpjpe(pred, gt) <= m.compute_mpjpe(pred, gt) + 1e-5


def test_pck_all_correct():
    gt = _rand()
    pred = gt + 0.001
    np.testing.assert_allclose(m.compute_pck(pred, gt, threshold=0.1), 1.0, atol=1e-6)


def test_pck_none_correct():
    gt = np.zeros((17, 3))
    pred = np.ones((17, 3)) * 10.0
    np.testing.assert_allclose(m.compute_pck(pred, gt, threshold=0.1), 0.0, atol=1e-6)


def test_jitter_zero_constant():
    seq = np.ones((50, 17, 3)) * 3.14
    np.testing.assert_allclose(m.compute_jitter_metric(seq), 0.0, atol=1e-8)


def test_jitter_positive_noisy():
    seq = np.random.default_rng(13).standard_normal((50, 17, 3))
    assert m.compute_jitter_metric(seq) > 0.0
