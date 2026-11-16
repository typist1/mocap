"""Tests for quaternion math."""
import math
import numpy as np
import pytest

anim = pytest.importorskip("src.animation.skeleton")
quat_multiply = anim.quat_multiply
quat_inverse = anim.quat_inverse
quat_slerp = anim.quat_slerp
rotation_from_directions = anim.rotation_from_directions
euler_from_quat = anim.euler_from_quat
quat_to_rotation_matrix = anim.quat_to_rotation_matrix
rotation_matrix_to_quat = anim.rotation_matrix_to_quat

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def _axisangle(axis, deg):
    ax = np.array(axis, dtype=float)
    ax /= np.linalg.norm(ax)
    r = math.radians(deg) / 2
    return np.array([*(ax * math.sin(r)), math.cos(r)])


def test_multiply_identity_right():
    q = _axisangle([1, 0, 0], 45)
    np.testing.assert_allclose(np.abs(quat_multiply(q, IDENTITY)), np.abs(q), atol=1e-7)


def test_multiply_identity_left():
    q = _axisangle([0, 1, 0], 90)
    np.testing.assert_allclose(np.abs(quat_multiply(IDENTITY, q)), np.abs(q), atol=1e-7)


def test_multiply_unit_norm():
    q1 = _axisangle([1, 0, 0], 30)
    q2 = _axisangle([0, 1, 0], 60)
    r = quat_multiply(q1, q2)
    np.testing.assert_allclose(np.linalg.norm(r), 1.0, atol=1e-6)


def test_inverse_times_self_identity():
    q = _axisangle([1, 2, 3], 55)
    prod = quat_multiply(q, quat_inverse(q))
    np.testing.assert_allclose(np.abs(prod), np.abs(IDENTITY), atol=1e-6)


def test_slerp_t0():
    q1 = _axisangle([1, 0, 0], 20)
    q2 = _axisangle([0, 1, 0], 80)
    np.testing.assert_allclose(np.abs(quat_slerp(q1, q2, 0.0)), np.abs(q1), atol=1e-6)


def test_slerp_t1():
    q1 = _axisangle([1, 0, 0], 20)
    q2 = _axisangle([0, 1, 0], 80)
    np.testing.assert_allclose(np.abs(quat_slerp(q1, q2, 1.0)), np.abs(q2), atol=1e-6)


def test_slerp_midpoint_unit():
    q1 = _axisangle([1, 0, 0], 10)
    q2 = _axisangle([0, 0, 1], 170)
    result = quat_slerp(q1, q2, 0.5)
    np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-6)


def test_rotation_from_directions_aligns():
    v_from = np.array([1.0, 0.0, 0.0])
    v_to = np.array([0.0, 1.0, 0.0])
    q = rotation_from_directions(v_from, v_to)
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-6)
    # Rotate v_from by q
    v_q = np.array([v_from[0], v_from[1], v_from[2], 0.0])
    tmp = quat_multiply(q, v_q)
    rotated = quat_multiply(tmp, quat_inverse(q))[:3]
    np.testing.assert_allclose(rotated, v_to, atol=1e-5)


def test_euler_identity():
    angles = euler_from_quat(IDENTITY)
    np.testing.assert_allclose(angles, [0.0, 0.0, 0.0], atol=1e-5)


def test_rotation_matrix_roundtrip():
    q = _axisangle([1, 1, 0], 60)
    R = quat_to_rotation_matrix(q)
    q2 = rotation_matrix_to_quat(R)
    # Quaternion double-cover: q and -q represent same rotation
    np.testing.assert_allclose(np.abs(q), np.abs(q2), atol=1e-5)
