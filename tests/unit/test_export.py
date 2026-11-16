"""Tests for animation export."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

export_mod = pytest.importorskip("src.animation.export")
skel_mod = pytest.importorskip("src.animation.skeleton")

BVHExporter = export_mod.BVHExporter
NPZExporter = export_mod.NPZExporter
JSONExporter = export_mod.JSONExporter
AnimationData = skel_mod.AnimationData
AnimationFrame = skel_mod.AnimationFrame
build_animation_skeleton = skel_mod.build_animation_skeleton

N_FRAMES = 10
FPS = 30.0


def _make_data() -> AnimationData:
    skel = build_animation_skeleton()
    frames = []
    for i in range(N_FRAMES):
        rots = {j: np.array([0.0, 0.0, 0.0, 1.0]) for j in skel.joints}
        frames.append(AnimationFrame(
            frame_index=i,
            timestamp=i / FPS,
            root_position=np.zeros(3),
            joint_rotations=rots,
        ))
    return AnimationData(frames=frames, skeleton=skel, fps=FPS)


@pytest.fixture
def data():
    return _make_data()


@pytest.fixture
def tmp():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def test_bvh_creates_file(data, tmp):
    path = BVHExporter().export(data, tmp / "test.bvh")
    assert path.exists() and path.stat().st_size > 0


def test_bvh_has_sections(data, tmp):
    path = BVHExporter().export(data, tmp / "test.bvh")
    content = path.read_text()
    assert "HIERARCHY" in content
    assert "MOTION" in content


def test_bvh_frame_count(data, tmp):
    path = BVHExporter().export(data, tmp / "test.bvh")
    lines = path.read_text().splitlines()
    frame_time_idx = next(i for i, l in enumerate(lines) if l.strip().startswith("Frame Time:"))
    data_lines = [l for l in lines[frame_time_idx + 1:] if l.strip()]
    assert len(data_lines) == N_FRAMES


def test_npz_creates_file(data, tmp):
    path = NPZExporter().export(data, tmp / "test.npz")
    assert path.exists()


def test_npz_loads_back(data, tmp):
    path = NPZExporter().export(data, tmp / "test.npz")
    loaded = np.load(str(path), allow_pickle=True)
    assert len(loaded.files) > 0


def test_json_creates_file(data, tmp):
    path = JSONExporter().export(data, tmp / "test.json")
    assert path.exists()


def test_json_backward_compatible(data, tmp):
    path = JSONExporter().export(data, tmp / "test.json")
    payload = json.loads(path.read_text())
    assert "fps" in payload
    assert "frames" in payload
    assert payload["fps"] == pytest.approx(FPS)
    assert len(payload["frames"]) == N_FRAMES
