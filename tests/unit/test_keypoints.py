"""Tests for keypoint definitions."""
import pytest

kp = pytest.importorskip("src.pose.keypoints")


def test_mp33_names_has_33():
    assert len(kp.MP33_NAMES) == 33


def test_mp33_names_unique():
    assert len(set(kp.MP33_NAMES)) == 33


def test_mp33_enum_has_33():
    assert len(kp.MP33) == 33


def test_mp33_indices_contiguous():
    assert sorted(int(v) for v in kp.MP33) == list(range(33))


def test_body_joints_valid():
    for idx in kp.BODY_JOINTS:
        assert 0 <= idx <= 32


def test_body_joints_no_duplicates():
    assert len(kp.BODY_JOINTS) == len(set(kp.BODY_JOINTS))


def test_skeleton_connections_valid():
    for a, b in kp.SKELETON_CONNECTIONS:
        assert 0 <= a <= 32
        assert 0 <= b <= 32
        assert a != b


def test_name_to_index_roundtrip():
    for name in kp.MP33_NAMES:
        idx = kp.name_to_index(name)
        assert kp.index_to_name(idx) == name


def test_name_to_index_raises():
    with pytest.raises(KeyError):
        kp.name_to_index("not_real")


def test_index_out_of_range():
    with pytest.raises(IndexError):
        kp.index_to_name(33)


def test_symmetric_pairs_valid():
    for left, right in kp.SYMMETRIC_PAIRS:
        assert 0 <= left <= 32
        assert 0 <= right <= 32
        assert left != right
        assert "left" in kp.index_to_name(left)
        assert "right" in kp.index_to_name(right)
