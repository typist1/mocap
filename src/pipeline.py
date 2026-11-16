"""Main processing pipeline: video → 3D pose → animation export."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.config.settings import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defensive module imports
# ---------------------------------------------------------------------------

try:
    from src.utils.validation import validate_video_path
except ImportError:
    def validate_video_path(p: Path) -> None:  # type: ignore[misc]
        if not p.exists():
            raise FileNotFoundError(f"Video not found: {p}")
        if not p.is_file():
            raise ValueError(f"Not a file: {p}")

try:
    from src.utils.logging import get_logger
except ImportError:
    def get_logger(name: str) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)

try:
    from src.video.metadata import VideoMetadata, extract_metadata
except Exception:  # pragma: no cover
    @dataclass
    class VideoMetadata:  # type: ignore[no-redef]
        width: int = 0
        height: int = 0
        fps: float = 30.0
        frame_count: int = 0
        duration_sec: float = 0.0

    def extract_metadata(path: Path) -> VideoMetadata:  # type: ignore[misc]
        import cv2
        cap = cv2.VideoCapture(str(path))
        meta = VideoMetadata(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        meta.duration_sec = meta.frame_count / meta.fps if meta.fps else 0.0
        cap.release()
        return meta

try:
    from src.video.loader import VideoLoader
except Exception:  # pragma: no cover
    class VideoLoader:  # type: ignore[no-redef]
        def __init__(self, path: Path, target_fps: float = 30.0, **_kw: Any) -> None:
            import cv2
            self._path = path
            self._target_fps = target_fps
            self._cap: Optional[cv2.VideoCapture] = None

        def __enter__(self) -> "VideoLoader":
            import cv2
            self._cap = cv2.VideoCapture(str(self._path))
            self._src_fps: float = self._cap.get(cv2.CAP_PROP_FPS) or self._target_fps
            self._frame_idx = 0
            self._next_time = 0.0
            return self

        def __exit__(self, *_: Any) -> None:
            if self._cap:
                self._cap.release()

        def __iter__(self):
            import cv2
            assert self._cap is not None
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    break
                t = self._frame_idx / self._src_fps
                self._frame_idx += 1
                if t + 1e-6 < self._next_time:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                while self._next_time <= t + 1e-6:
                    yield rgb
                    self._next_time += 1.0 / self._target_fps

try:
    from src.video.preprocessing import FramePreprocessor
except Exception:  # pragma: no cover
    class FramePreprocessor:  # type: ignore[no-redef]
        def __init__(self, max_dimension: int = 1280, **_kw: Any) -> None:
            self._max_dim = max_dimension

        def process(self, frame: np.ndarray) -> np.ndarray:
            h, w = frame.shape[:2]
            if max(h, w) <= self._max_dim:
                return frame
            scale = self._max_dim / max(h, w)
            import cv2
            return cv2.resize(frame, (int(w * scale), int(h * scale)))

# pose 2D
try:
    from src.pose.pose_2d import MediaPipePose2D
except Exception:  # pragma: no cover
    MediaPipePose2D = None  # type: ignore[assignment,misc]

# filtering
try:
    from src.filtering.temporal_filter import LandmarkFilter
except Exception:  # pragma: no cover
    LandmarkFilter = None  # type: ignore[assignment,misc]

# camera
try:
    from src.reconstruction.camera import estimate_focal_length, make_camera
    _CAMERA_AVAILABLE = True
except Exception:  # pragma: no cover
    _CAMERA_AVAILABLE = False
    def make_camera(width: int, height: int, focal_px: Optional[float] = None) -> None:  # type: ignore[misc]
        return None
    def estimate_focal_length(*_a: Any, **_kw: Any) -> float:  # type: ignore[misc]
        return 0.0

# optimizer
try:
    from src.reconstruction.optimization import PoseOptimizer
except Exception:  # pragma: no cover
    PoseOptimizer = None  # type: ignore[assignment,misc]

# animation skeleton
try:
    from src.animation.skeleton import (
        AnimationData,
        AnimationFrame,
        AnimationSkeleton,
        build_animation_skeleton,
    )
    _SKELETON_AVAILABLE = True
except Exception:  # pragma: no cover
    _SKELETON_AVAILABLE = False
    AnimationSkeleton = None  # type: ignore[assignment,misc]
    AnimationData = None      # type: ignore[assignment,misc]
    AnimationFrame = None     # type: ignore[assignment,misc]
    build_animation_skeleton = None  # type: ignore[assignment,misc]

# retargeting
try:
    from src.animation.retargeting import MediaPipeToSkeletonRetargeter
except Exception:  # pragma: no cover
    MediaPipeToSkeletonRetargeter = None  # type: ignore[assignment,misc]

# export
try:
    from src.animation.export import AnimationExporter
    _EXPORTER_AVAILABLE = True
except Exception:  # pragma: no cover
    _EXPORTER_AVAILABLE = False
    AnimationExporter = None  # type: ignore[assignment,misc]

# metrics
try:
    from src.evaluation.metrics import MetricsReporter
    _METRICS_AVAILABLE = True
except Exception:  # pragma: no cover
    _METRICS_AVAILABLE = False
    MetricsReporter = None  # type: ignore[assignment,misc]

# tqdm
try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except Exception:
    _TQDM_AVAILABLE = False


def _progress(iterable, **kwargs):
    if _TQDM_AVAILABLE:
        return _tqdm(iterable, **kwargs)
    return iterable


def _estimate_body_proportions(sequence_3d: np.ndarray, n_frames: int = 0) -> dict:
    """Compute median bone lengths from SKELETON_CONNECTIONS. Returns {bone_name: length_m}."""
    try:
        from src.pose.keypoints import SKELETON_CONNECTIONS, MP33_NAMES
    except Exception:
        return {}
    seq = sequence_3d[:n_frames] if 0 < n_frames <= len(sequence_3d) else sequence_3d
    valid = ~np.any(np.isnan(seq.reshape(seq.shape[0], -1)), axis=1)
    seq = seq[valid]
    if len(seq) == 0:
        return {}
    result: dict = {}
    for a, b in SKELETON_CONNECTIONS:
        if a >= seq.shape[1] or b >= seq.shape[1]:
            continue
        lengths = np.linalg.norm(seq[:, b] - seq[:, a], axis=1)
        result[f"{MP33_NAMES[a]}_{MP33_NAMES[b]}"] = float(np.median(lengths))
    return result


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProcessingResult:
    source_video: Path
    output_files: Dict[str, Path] = field(default_factory=dict)
    metrics: Optional[object] = None
    processing_time_sec: float = 0.0
    frame_count: int = 0
    detection_rate: float = 0.0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class MocapPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._log = get_logger(__name__)

    def process(
        self,
        video_path: Path,
        output_dir: Optional[Path] = None,
    ) -> ProcessingResult:
        pipeline_start = time.perf_counter()

        video_path = Path(video_path)
        if output_dir is None:
            output_dir = Path(self.settings.animation.output_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = video_path.stem

        # ---- 1. Validate ------------------------------------------------
        self._log.info("Stage 1/8: Validating inputs")
        validate_video_path(video_path)

        # ---- 2. Metadata ------------------------------------------------
        self._log.info("Stage 2/8: Extracting video metadata")
        meta = extract_metadata(video_path)
        self._log.info(
            "  %dx%d @ %.2f fps, %d frames, %.1fs",
            meta.width, meta.height, meta.fps, meta.frame_count, meta.duration_sec,
        )
        if meta.frame_count == 0:
            raise RuntimeError(f"Video appears to have no frames: {video_path}")

        # ---- 3. Initialize components -----------------------------------
        self._log.info("Stage 3/8: Initializing components")
        preprocessor = FramePreprocessor(max_dimension=self.settings.video.max_dimension)
        pose2d = self._build_pose2d_estimator()
        landmark_filter = self._build_landmark_filter(meta.fps)

        # ---- 4. Per-frame estimation ------------------------------------
        self._log.info("Stage 4/8: Running per-frame pose estimation")
        world_seq: List[np.ndarray] = []
        image_seq: List[np.ndarray] = []
        empty_lm = np.full((33, 3), np.nan, dtype=np.float32)
        detected_count = 0
        frame_index = 0

        with VideoLoader(meta, self.settings.video, preprocessor) as loader:
            frame_iter = _progress(
                loader.frames(),
                desc="Estimating pose",
                unit="frame",
                total=meta.frame_count if meta.frame_count > 0 else None,
                dynamic_ncols=True,
            )
            for fd in frame_iter:
                lm_world, lm_image = self._estimate_frame(pose2d, fd.frame_rgb, frame_index)
                frame_index += 1

                if lm_world is not None:
                    if landmark_filter is not None:
                        vis = lm_image[:, 2] if lm_image is not None else None
                        lm_world = landmark_filter.filter(lm_world, visibility=vis)
                    world_seq.append(lm_world)
                    image_seq.append(lm_image if lm_image is not None else empty_lm.copy())
                    detected_count += 1
                else:
                    world_seq.append(empty_lm.copy())
                    image_seq.append(empty_lm.copy())

        frame_count = len(world_seq)
        self._log.info(
            "  %d frames collected, detection in %d (%.1f%%)",
            frame_count, detected_count,
            100.0 * detected_count / frame_count if frame_count else 0.0,
        )

        if frame_count == 0:
            return ProcessingResult(
                source_video=video_path,
                processing_time_sec=time.perf_counter() - pipeline_start,
                frame_count=0,
                detection_rate=0.0,
            )

        if detected_count == 0:
            self._log.warning("No person detected in %s", video_path.name)
            return ProcessingResult(
                source_video=video_path,
                processing_time_sec=time.perf_counter() - pipeline_start,
                frame_count=frame_count,
                detection_rate=0.0,
            )

        detection_rate = detected_count / frame_count

        # ---- 5. Post-process pose sequence ------------------------------
        self._log.info("Stage 5/8: Post-processing pose sequence")
        world_arr = np.stack(world_seq, axis=0).astype(np.float32)   # (F, 33, 3)
        image_arr = np.stack(image_seq, axis=0).astype(np.float32)   # (F, 33, 3)

        body_props = _estimate_body_proportions(
            world_arr,
            n_frames=self.settings.reconstruction.bone_estimation_frames,
        )
        self._log.debug("  body proportions: %d bones estimated", len(body_props))

        world_arr = self._optimize(world_arr, image_arr)

        # ---- 6. Camera model --------------------------------------------
        self._log.info("Stage 6/8: Estimating camera model")
        camera = self._estimate_camera(world_arr, image_arr, meta)

        # ---- 7. Build animation skeleton + retarget ---------------------
        self._log.info("Stage 7/8: Building skeleton and retargeting")
        anim_skeleton = build_animation_skeleton(body_props) if build_animation_skeleton else None
        anim_data = self._retarget(anim_skeleton, world_arr, str(video_path))

        # ---- 8. Export --------------------------------------------------
        self._log.info("Stage 8/8: Exporting animation")
        output_files = self._export(anim_data, output_dir, stem)
        self._log.info("  exported: %s", list(output_files.keys()))

        # ---- Metrics ----------------------------------------------------
        metrics = self._compute_metrics(world_arr, camera, output_dir, stem)

        processing_time = time.perf_counter() - pipeline_start
        self._log.info(
            "Pipeline complete in %.2fs (%.1f fps throughput)",
            processing_time,
            frame_count / processing_time if processing_time > 0 else 0.0,
        )

        return ProcessingResult(
            source_video=video_path,
            output_files=output_files,
            metrics=metrics,
            processing_time_sec=processing_time,
            frame_count=frame_count,
            detection_rate=detection_rate,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_pose2d_estimator(self) -> Any:
        if MediaPipePose2D is None:
            self._log.warning("pose.pose_2d not available — using built-in fallback")
            return _FallbackPose2D(self.settings)
        try:
            estimator = MediaPipePose2D(
                pose_config=self.settings.pose,
                detection_config=self.settings.detection,
            )
            self._log.debug("MediaPipePose2D initialized")
            return estimator
        except Exception as exc:
            self._log.warning("MediaPipePose2D init failed (%s); using fallback", exc)
            return _FallbackPose2D(self.settings)

    def _build_landmark_filter(self, source_fps: float) -> Any:
        if LandmarkFilter is None:
            return None
        try:
            return LandmarkFilter(
                n_landmarks=33,
                freq=source_fps,
                min_cutoff=self.settings.filtering.one_euro_min_cutoff,
                beta=self.settings.filtering.one_euro_beta,
                d_cutoff=self.settings.filtering.one_euro_d_cutoff,
            )
        except Exception as exc:
            self._log.warning("LandmarkFilter init failed (%s) — skipping", exc)
            return None

    def _estimate_frame(
        self,
        pose2d: Any,
        frame: np.ndarray,
        frame_index: int = 0,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            result = pose2d.estimate(frame, frame_index)
        except Exception as exc:
            self._log.debug("Pose estimation error: %s", exc)
            return None, None

        if result is None:
            return None, None

        # Pose2DResult from src.pose.pose_2d
        if hasattr(result, "landmarks_world") and hasattr(result, "landmarks_image"):
            return result.landmarks_world, result.landmarks_image

        # Fallback: 2-tuple (world, image)
        if isinstance(result, (tuple, list)) and len(result) == 2:
            return result[0], result[1]

        self._log.warning("Unexpected pose result type: %s", type(result))
        return None, None

    def _optimize(self, world_arr: np.ndarray, image_arr: np.ndarray) -> np.ndarray:
        if PoseOptimizer is None:
            return world_arr
        try:
            optimizer = PoseOptimizer(
                filtering_config=self.settings.filtering,
                reconstruction_config=self.settings.reconstruction,
            )
            visibility = image_arr[:, :, 2]  # [F, 33]
            return optimizer.optimize(
                world_arr,
                visibility,
                fps=self.settings.video.target_fps,
                body_proportions=None,  # bone normalization skipped (no kinematics Skeleton)
                skeleton=None,
            )
        except Exception as exc:
            self._log.warning("PoseOptimizer failed (%s) — skipping optimization", exc)
            return world_arr

    def _estimate_camera(self, world_arr: np.ndarray, image_arr: np.ndarray, meta: Any) -> Any:
        if not _CAMERA_AVAILABLE:
            return None
        try:
            first_valid = self._first_valid_frame(world_arr)
            focal_length = self.settings.reconstruction.focal_length
            if focal_length == "auto" and first_valid is not None:
                focal_px = estimate_focal_length(
                    image_arr[first_valid],
                    world_arr[first_valid],
                    meta.width,
                    meta.height,
                )
            elif focal_length != "auto":
                focal_px = float(focal_length)
            else:
                focal_px = 1.2 * max(meta.width, meta.height)
            return make_camera(meta.width, meta.height, focal_px)
        except Exception as exc:
            self._log.warning("Camera estimation failed (%s) — using default", exc)
            return make_camera(meta.width, meta.height)

    def _retarget(
        self,
        skeleton: Any,
        world_arr: np.ndarray,
        source_video: str = "",
    ) -> Any:
        if MediaPipeToSkeletonRetargeter is None or skeleton is None:
            self._log.debug("Retargeting not available — returning stub animation data")
            return _make_stub_animation_data(skeleton, world_arr, self.settings.video.target_fps, source_video)
        try:
            retargeter = MediaPipeToSkeletonRetargeter()
            root_motion = np.zeros((world_arr.shape[0], 3), dtype=np.float32)
            return retargeter.retarget_sequence(
                world_arr,
                root_motion,
                fps=self.settings.video.target_fps,
                skeleton=skeleton,
                source_video=source_video,
            )
        except Exception as exc:
            self._log.warning("Retargeting failed (%s) — using stub", exc)
            return _make_stub_animation_data(skeleton, world_arr, self.settings.video.target_fps, source_video)

    def _export(self, anim_data: Any, output_dir: Path, stem: str) -> Dict[str, Path]:
        if AnimationExporter is None or anim_data is None:
            self._log.warning("AnimationExporter not available — no files written")
            return {}
        try:
            orig_dir = self.settings.animation.output_dir
            self.settings.animation.output_dir = str(output_dir)
            exporter = AnimationExporter(self.settings.animation)
            result = exporter.export_all(anim_data, stem)
            self.settings.animation.output_dir = orig_dir
            return result
        except Exception as exc:
            self._log.error("Export failed: %s", exc)
            return {}

    def _compute_metrics(
        self,
        world_arr: np.ndarray,
        camera: Any,
        output_dir: Path,
        stem: str,
    ) -> Optional[Any]:
        if MetricsReporter is None:
            return None
        try:
            reporter = MetricsReporter()
            metrics = reporter.compute_all(
                sequence_3d=world_arr,
                sequence_2d_px=None,
                camera=camera,
                fps=self.settings.video.target_fps,
                ground_truth=None,
            )
            if self.settings.evaluation.save_metrics_csv:
                reporter.save_csv(metrics, output_dir / f"{stem}_metrics.csv")
            return metrics
        except Exception as exc:
            self._log.warning("Metrics computation failed (%s)", exc)
            return None

    @staticmethod
    def _first_valid_frame(arr: np.ndarray) -> Optional[int]:
        for i, frame in enumerate(arr):
            if not np.any(np.isnan(frame)):
                return i
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_stub_animation_data(skeleton: Any, world_arr: np.ndarray, fps: float, source_video: str = "") -> Any:
    """Build minimal AnimationData when retargeting is unavailable."""
    if AnimationFrame is None or AnimationData is None:
        return None
    frames = []
    identity_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    joint_names = skeleton.joints if skeleton is not None else []
    for f in range(len(world_arr)):
        pts = world_arr[f]
        l_hip, r_hip = pts[23], pts[24]
        if np.any(np.isnan(l_hip)) or np.any(np.isnan(r_hip)):
            root_pos = np.zeros(3, dtype=np.float32)
        else:
            root_pos = ((l_hip + r_hip) / 2.0).astype(np.float32)
        frames.append(AnimationFrame(
            frame_index=f,
            timestamp=f / fps,
            root_position=root_pos,
            joint_rotations={j: identity_q.copy() for j in joint_names},
        ))
    return AnimationData(frames=frames, skeleton=skeleton, fps=fps, source_video=source_video)


class _FallbackPose2D:
    """Minimal MediaPipe-based pose estimator used when pose_2d module is absent."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._landmarker = None
        self._prev_rot = None
        self._rotations = None
        self._init_landmarker()

    def _init_landmarker(self) -> None:
        import urllib.request

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
        except Exception:
            logger.error("mediapipe not available — pose estimation unavailable")
            return

        model_url = (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        )
        model_path = Path(__file__).parent.parent / "pose_landmarker_heavy.task"
        if not model_path.exists():
            logger.info("Downloading MediaPipe pose model (one-time)...")
            urllib.request.urlretrieve(model_url, str(model_path))

        import cv2
        self._rotations = [
            None,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_180,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        ]

        options = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=self._settings.detection.min_detection_confidence,
            min_pose_presence_confidence=self._settings.detection.min_detection_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._mp = mp
        logger.debug("Fallback MediaPipe landmarker initialized from %s", model_path)

    def estimate(self, frame: np.ndarray, frame_index: int = 0):
        if self._landmarker is None:
            return None

        import cv2
        BODY = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        threshold = self._settings.detection.rotation_fallback_threshold
        min_vis = self._settings.detection.min_visibility

        def _detect_one(rgb, rot):
            img = rgb if rot is None else cv2.rotate(rgb, rot)
            result = self._landmarker.detect(
                self._mp.Image(
                    image_format=self._mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(img),
                )
            )
            if not result.pose_landmarks:
                return None
            img_pts = np.array(
                [[p.x, p.y, p.visibility] for p in result.pose_landmarks[0]],
                dtype=np.float32,
            )
            world_pts = np.array(
                [[p.x, p.y, p.z] for p in result.pose_world_landmarks[0]],
                dtype=np.float32,
            )
            if rot == cv2.ROTATE_180:
                img_pts[:, 0] = 1 - img_pts[:, 0]
                img_pts[:, 1] = 1 - img_pts[:, 1]
                world_pts[:, 0] = -world_pts[:, 0]
                world_pts[:, 1] = -world_pts[:, 1]
            elif rot == cv2.ROTATE_90_CLOCKWISE:
                img_pts[:, 0], img_pts[:, 1] = img_pts[:, 1].copy(), 1 - img_pts[:, 0].copy()
                world_pts[:, 0], world_pts[:, 1] = world_pts[:, 1].copy(), -world_pts[:, 0].copy()
            elif rot == cv2.ROTATE_90_COUNTERCLOCKWISE:
                img_pts[:, 0], img_pts[:, 1] = 1 - img_pts[:, 1].copy(), img_pts[:, 0].copy()
                world_pts[:, 0], world_pts[:, 1] = -world_pts[:, 1].copy(), world_pts[:, 0].copy()
            score = float(img_pts[BODY, 2].mean())
            return score, img_pts, world_pts

        candidates = {}
        candidates[self._prev_rot] = _detect_one(frame, self._prev_rot)
        first = candidates[self._prev_rot]

        if first is None or first[0] < threshold:
            for rot in (self._rotations or [None]):
                if rot not in candidates:
                    candidates[rot] = _detect_one(frame, rot)

        best_rot, best = self._prev_rot, first
        for rot, res in candidates.items():
            if res is None:
                continue
            if best is None or res[0] > best[0] + 0.1:
                best_rot, best = rot, res

        self._prev_rot = best_rot
        if best is None:
            return None

        _, img_pts, world_pts = best
        low_vis = img_pts[:, 2] < min_vis
        world_pts[low_vis] = np.nan
        return world_pts, img_pts
