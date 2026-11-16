"""Tests for temporal filters and smoothing."""
import numpy as np
import pytest

tf = pytest.importorskip("src.filtering.temporal_filter")
sm = pytest.importorskip("src.filtering.smoothing")

OneEuroFilter = tf.OneEuroFilter
OneEuroFilter3D = tf.OneEuroFilter3D
KalmanFilter3D = tf.KalmanFilter3D
LandmarkFilter = tf.LandmarkFilter


def test_one_euro_constant():
    f = OneEuroFilter(freq=30.0, min_cutoff=1.5, beta=0.1)
    out = [f.filter(5.0) for _ in range(60)]
    np.testing.assert_allclose(out[20:], 5.0, atol=1e-5)


def test_one_euro_reduces_noise():
    rng = np.random.default_rng(99)
    noise = rng.standard_normal(200) * 0.5
    f = OneEuroFilter(freq=30.0, min_cutoff=1.5, beta=0.05)
    filtered = np.array([f.filter(3.0 + n) for n in noise])
    assert filtered[20:].std() < noise[20:].std()


def test_one_euro_3d_shape():
    f = OneEuroFilter3D(freq=30.0)
    result = f.filter(np.array([1.0, 2.0, 3.0]))
    assert result.shape == (3,)


def test_kalman_converges():
    true_pos = np.array([1.0, 2.0, 3.0])
    kf = KalmanFilter3D(process_noise=1e-3, measurement_noise=1e-2)
    rng = np.random.default_rng(0)
    for _ in range(100):
        est = kf.update(true_pos + rng.standard_normal(3) * 0.05)
    assert np.linalg.norm(est - true_pos) < 0.1


def test_kalman_no_diverge():
    kf = KalmanFilter3D()
    for _ in range(30):
        kf.update(np.array([0.5, 0.5, 0.5]) + np.random.randn(3) * 0.02)
    preds = [kf.predict() for _ in range(30)]
    assert all(np.linalg.norm(p - 0.5) < 1.0 for p in preds)


def test_landmark_filter_shape():
    lf = LandmarkFilter(n_landmarks=33, freq=30.0)
    pts = np.random.randn(33, 3).astype(np.float32)
    result = lf.filter(pts)
    assert result.shape == (33, 3)


def test_savgol_shape():
    seq = np.random.randn(50, 33, 3)
    result = sm.savitzky_golay_smooth(seq, window=11, polyorder=3)
    assert result.shape == seq.shape


def test_savgol_too_short():
    seq = np.random.randn(5, 33, 3)
    result = sm.savitzky_golay_smooth(seq, window=11, polyorder=3)
    np.testing.assert_array_equal(result, seq)


def test_fill_gaps_no_nans():
    seq = np.random.randn(50, 33, 3)
    seq[10:15, 5, :] = np.nan
    result = sm.fill_temporal_gaps(seq)
    assert not np.any(np.isnan(result))


def test_fill_gaps_linear():
    seq = np.zeros((10, 1, 1))
    seq[:, 0, 0] = np.arange(10, dtype=float)
    seq[3:6, 0, 0] = np.nan
    result = sm.fill_temporal_gaps(seq)
    np.testing.assert_allclose(result[4, 0, 0], 4.0, atol=0.1)


def test_jitter_zero_constant():
    seq = np.ones((50, 33, 3)) * 2.5
    assert sm.compute_jitter(seq) == pytest.approx(0.0, abs=1e-8)


def test_jitter_positive_noisy():
    rng = np.random.default_rng(7)
    seq = rng.standard_normal((50, 33, 3))
    assert sm.compute_jitter(seq) > 0.0
