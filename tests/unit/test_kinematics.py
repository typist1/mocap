"""Tests for skeleton and kinematics."""
import numpy as np
import pytest

kin = pytest.importorskip("src.reconstruction.kinematics")
kp = pytest.importorskip("src.pose.keypoints")
MP33 = kp.MP33


def _tpose():
    pose = np.zeros((33, 3))
    pose[MP33.LEFT_HIP] = [-0.10, 0.90, 0.0]
    pose[MP33.RIGHT_HIP] = [0.10, 0.90, 0.0]
    pose[MP33.LEFT_KNEE] = [-0.10, 0.50, 0.0]
    pose[MP33.RIGHT_KNEE] = [0.10, 0.50, 0.0]
    pose[MP33.LEFT_ANKLE] = [-0.10, 0.10, 0.0]
    pose[MP33.RIGHT_ANKLE] = [0.10, 0.10, 0.0]
    pose[MP33.LEFT_SHOULDER] = [-0.20, 1.40, 0.0]
    pose[MP33.RIGHT_SHOULDER] = [0.20, 1.40, 0.0]
    pose[MP33.LEFT_ELBOW] = [-0.40, 1.10, 0.0]
    pose[MP33.RIGHT_ELBOW] = [0.40, 1.10, 0.0]
    pose[MP33.LEFT_WRIST] = [-0.55, 0.85, 0.0]
    pose[MP33.RIGHT_WRIST] = [0.55, 0.85, 0.0]
    pose[MP33.NOSE] = [0.0, 1.70, 0.0]
    pose[MP33.LEFT_FOOT_INDEX] = [-0.10, 0.0, 0.1]
    pose[MP33.RIGHT_FOOT_INDEX] = [0.10, 0.0, 0.1]
    rng = np.random.default_rng(0)
    for i in range(33):
        if np.all(pose[i] == 0):
            pose[i] = rng.uniform(-0.1, 0.1, 3)
    return pose


def test_skeleton_has_joints():
    skel = kin.build_mediapipe_skeleton()
    names = set(skel.joints.keys())
    assert any("hip" in n for n in names)
    assert any("knee" in n for n in names)
    assert any("shoulder" in n for n in names)


def test_bone_lengths_positive():
    skel = kin.build_mediapipe_skeleton()
    pose = _tpose()
    lengths = skel.compute_bone_lengths(pose)
    assert len(lengths) > 0
    for name, length in lengths.items():
        assert length > 0, f"Bone '{name}' has non-positive length {length}"


def test_bone_normalization_preserves_direction():
    skel = kin.build_mediapipe_skeleton()
    pose = _tpose()
    target = skel.compute_bone_lengths(pose)
    rng = np.random.default_rng(42)
    perturbed = pose + rng.standard_normal(pose.shape) * 0.05
    normalized = skel.normalize_bone_lengths(perturbed, target)
    achieved = skel.compute_bone_lengths(normalized)
    for name, tl in target.items():
        if name in achieved:
            assert abs(achieved[name] - tl) < 1e-3, f"Bone {name}: target={tl}, got={achieved[name]}"
