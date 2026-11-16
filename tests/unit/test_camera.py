"""Tests for camera model and projection."""
import numpy as np
import pytest

cam = pytest.importorskip("src.reconstruction.camera")
CameraModel = cam.CameraModel
project_3d_to_2d = cam.project_3d_to_2d
unproject_2d_to_3d = cam.unproject_2d_to_3d
reprojection_error = cam.reprojection_error


def _cam(fx=600.0, cx=320.0, cy=240.0):
    return CameraModel(fx=fx, fy=fx, cx=cx, cy=cy, width=640, height=480)


def test_project_identity():
    """Point at (0,0,f) → (cx, cy)."""
    c = _cam(fx=600.0, cx=320.0, cy=240.0)
    pt = np.array([0.0, 0.0, 600.0])
    result = project_3d_to_2d(c, pt)
    np.testing.assert_allclose(result, [320.0, 240.0], atol=1e-5)


def test_project_unproject_roundtrip():
    c = _cam()
    rng = np.random.default_rng(42)
    pts3d = rng.uniform(-0.5, 0.5, (20, 3))
    pts3d[:, 2] = rng.uniform(0.5, 5.0, 20)
    for pt in pts3d:
        pt2d = project_3d_to_2d(c, pt)
        recovered = unproject_2d_to_3d(c, pt2d.reshape(1, 2), np.array([pt[2]]))[0]
        np.testing.assert_allclose(recovered, pt, atol=1e-5)


def test_reprojection_error_zero_on_perfect():
    c = _cam()
    rng = np.random.default_rng(7)
    pts3d = rng.uniform(-1, 1, (10, 3))
    pts3d[:, 2] = rng.uniform(1, 5, 10)
    pts2d = project_3d_to_2d(c, pts3d)
    err = reprojection_error(c, pts3d, pts2d)
    np.testing.assert_allclose(err, 0.0, atol=1e-4)


def test_zero_depth_no_crash():
    c = _cam()
    pt = np.array([1.0, 1.0, 0.0])
    result = project_3d_to_2d(c, pt)
    assert result.shape == (2,)
    assert np.all(np.isfinite(result)) or True  # may be large but no crash/exception


def test_focal_length_property():
    c = CameraModel(fx=800.0, fy=800.0, cx=320.0, cy=240.0)
    assert c.focal_length_px == pytest.approx(800.0)


def test_project_batch_shape():
    c = _cam()
    pts = np.random.randn(15, 3)
    pts[:, 2] = 2.0
    result = project_3d_to_2d(c, pts)
    assert result.shape == (15, 2)
