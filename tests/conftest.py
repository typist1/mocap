"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def dummy_settings():
    mod = pytest.importorskip("src.config.settings")
    return mod.Settings()


@pytest.fixture
def dummy_frame_rgb():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def dummy_landmarks_33():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0.1, 0.9, size=(33, 2)).astype(np.float32)
    vis = rng.uniform(0.5, 1.0, size=(33, 1)).astype(np.float32)
    return np.concatenate([xy, vis], axis=1)


@pytest.fixture
def dummy_sequence_3d():
    rng = np.random.default_rng(1)
    return (rng.standard_normal((50, 33, 3)) * 0.5).astype(np.float64)


@pytest.fixture
def dummy_visibility():
    return np.full((50, 33), 0.9, dtype=np.float32)
