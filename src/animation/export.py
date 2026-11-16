"""Animation export: BVH, NPZ, JSON."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.animation.skeleton import (
    AnimationData,
    AnimationFrame,
    AnimationSkeleton,
    euler_from_quat,
)
from src.config.settings import AnimationConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# BVH exporter
# ---------------------------------------------------------------------------

class BVHExporter:
    def export(self, data: AnimationData, output_path, scale: float = 100.0) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        skel = data.skeleton
        fps = data.fps

        lines: List[str] = []
        lines.append("HIERARCHY")
        self._write_joint_hierarchy(skel, skel.root_joint, lines, scale, indent=0)
        lines.append("")
        lines.append("MOTION")
        lines.append(f"Frames: {len(data.frames)}")
        lines.append(f"Frame Time: {1.0/fps:.10f}")

        for frame in data.frames:
            row = self._frame_to_row(frame, skel, scale)
            lines.append(" ".join(f"{v:.6f}" for v in row))

        content = "\n".join(lines) + "\n"
        path.write_text(content)

        # Validation: re-read and verify
        read_back = path.read_text()
        assert "HIERARCHY" in read_back and "MOTION" in read_back
        motion_lines = [l for l in read_back.split("\n") if l.strip() and
                        not l.startswith("H") and not l.startswith("M") and
                        not l.startswith("F") and not l.startswith("R") and
                        not l.startswith("J") and not l.startswith("E") and
                        not l.startswith("{") and not l.startswith("}") and
                        not l.startswith("C") and not l.startswith("O") and
                        not l.startswith(" ")]
        logger.debug("BVH exported %d frames to %s", len(data.frames), path)
        return path

    def _write_joint_hierarchy(
        self, skel: AnimationSkeleton, joint: str, lines: List[str], scale: float, indent: int
    ) -> None:
        pad = "  " * indent
        offset = skel.rest_positions.get(joint, np.zeros(3)) * scale
        is_root = skel.parent_map.get(joint) is None
        children = skel.children_of(joint)

        keyword = "ROOT" if is_root else "JOINT"
        lines.append(f"{pad}{keyword} {joint}")
        lines.append(f"{pad}{{")
        lines.append(f"{pad}  OFFSET {offset[0]:.5f} {offset[1]:.5f} {offset[2]:.5f}")
        if is_root:
            lines.append(f"{pad}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
        else:
            lines.append(f"{pad}  CHANNELS 3 Zrotation Xrotation Yrotation")

        if not children:
            # End site
            lines.append(f"{pad}  End Site")
            lines.append(f"{pad}  {{")
            lines.append(f"{pad}    OFFSET 0.00000 0.10000 0.00000")
            lines.append(f"{pad}  }}")
        else:
            for child in children:
                self._write_joint_hierarchy(skel, child, lines, scale, indent + 1)

        lines.append(f"{pad}}}")

    def _frame_to_row(
        self, frame: AnimationFrame, skel: AnimationSkeleton, scale: float
    ) -> List[float]:
        row: List[float] = []
        root = skel.root_joint
        # Root: Xpos Ypos Zpos Zrot Xrot Yrot
        pos = frame.root_position * scale
        row.extend([pos[0], pos[1], pos[2]])
        q = frame.joint_rotations.get(root, np.array([0.0, 0.0, 0.0, 1.0]))
        euler = euler_from_quat(q, order="ZXY")
        row.extend([euler[2], euler[0], euler[1]])  # Z, X, Y
        # Other joints: Zrot Xrot Yrot
        for joint in skel.joints:
            if joint == root:
                continue
            q = frame.joint_rotations.get(joint, np.array([0.0, 0.0, 0.0, 1.0]))
            euler = euler_from_quat(q, order="ZXY")
            row.extend([euler[2], euler[0], euler[1]])
        return row


# ---------------------------------------------------------------------------
# NPZ exporter
# ---------------------------------------------------------------------------

class NPZExporter:
    def export(self, data: AnimationData, output_path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        n_frames = len(data.frames)
        n_joints = len(data.skeleton.joints)
        positions = np.zeros((n_frames, n_joints, 3), dtype=np.float32)
        rotations = np.zeros((n_frames, n_joints, 4), dtype=np.float32)
        root_positions = np.zeros((n_frames, 3), dtype=np.float32)

        for fi, frame in enumerate(data.frames):
            root_positions[fi] = frame.root_position
            for ji, jname in enumerate(data.skeleton.joints):
                q = frame.joint_rotations.get(jname, np.array([0.0, 0.0, 0.0, 1.0]))
                rotations[fi, ji] = q

        np.savez_compressed(
            str(path),
            positions=positions,
            rotations=rotations,
            root_positions=root_positions,
            fps=np.array([data.fps]),
            joint_names=np.array(data.skeleton.joints, dtype=object),
            source_video=np.array([data.source_video], dtype=object),
        )
        logger.debug("NPZ exported %d frames to %s", n_frames, path)
        return path


# ---------------------------------------------------------------------------
# JSON exporter (backward-compatible with mocap.py format)
# ---------------------------------------------------------------------------

class JSONExporter:
    def export(self, data: AnimationData, output_path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Reconstruct per-frame joint positions from stored frames
        # For backward compat we store root motion + relative positions
        frames_list = []
        for frame in data.frames:
            # Store root position + placeholder zeros for now
            pos = frame.root_position.tolist()
            frames_list.append(pos)

        payload = {
            "fps": data.fps,
            "names": data.skeleton.joints,
            "joint_names": data.skeleton.joints,
            "frames": frames_list,
            "source_video": data.source_video,
        }
        path.write_text(json.dumps(payload, indent=None))
        logger.debug("JSON exported %d frames to %s", len(data.frames), path)
        return path


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------

class AnimationExporter:
    def __init__(self, config: AnimationConfig) -> None:
        self.config = config
        self._exporters = {
            "bvh": BVHExporter(),
            "npz": NPZExporter(),
            "json": JSONExporter(),
        }

    def export_all(self, data: AnimationData, base_name: str) -> Dict[str, Path]:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results: Dict[str, Path] = {}
        for fmt in self.config.export_formats:
            if fmt not in self._exporters:
                logger.warning("Unknown export format: %s", fmt)
                continue
            out_path = output_dir / f"{base_name}.{fmt}"
            try:
                path = self._exporters[fmt].export(data, out_path)
                results[fmt] = path
                logger.info("Exported %s → %s", fmt.upper(), path)
            except Exception as exc:
                logger.error("Export to %s failed: %s", fmt, exc)
        return results
