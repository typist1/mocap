"""Main processing pipeline: video → 3D pose → animation export.

Orchestrates every stage of the mocap system in a single Process() call.
Defensive imports are used throughout; modules that do not yet exist are
silently substituted with None-returning stubs so the file can be imported
even on a partial codebase.
"""
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

# utils
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

# video
try:
    from src.video.metadata import VideoMetadata, extract_metadata
except ImportError:  # pragma: no cover
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
except ImportError:  # pragma: no cover
    class VideoLoader:  # type: ignore[no-redef]
        """Minimal fallback loader using raw OpenCV."""
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
except ImportError:  # pragma: no cover
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
except ImportError:  # pragma: no cover
    MediaPipePose2D = None  # type: ignore[assignment,misc]

# filtering
try:
    from src.filtering.temporal_filter import LandmarkFilter
except ImportError:  # pragma: no cover
    LandmarkFilter = None  # type: ignore[assignment,misc]

# reconstruction
try:
    from src.reconstruction.camera import estimate_camera
except ImportError:  # pragma: no cover
    def estimate_camera(world_pts: np.ndarray, **_kw: Any) -> Any:  # type: ignore[misc]
        return None

try:
    from src.reconstruction.kinematics import estimate_body_proportions
except ImportError:  # pragma: no cover
    def estimate_body_proportions(seq: np.ndarray, **_kw: Any) -> Any:  # type: ignore[misc]
        return {}

try:
    from src.reconstruction.optimization import PoseOptimizer
except ImportError:  # pragma: no cover
    class PoseOptimizer:  # type: ignore[no-redef]
        def __init__(self, **_kw: Any) -> None:
            pass

        def run(self, seq: np.ndarray, **_kw: Any) -> np.ndarray:
            return seq

# animation
try:
    from src.animation.skeleton import AnimationSkeleton
except ImportError:  # pragma: no cover
    AnimationSkeleton = None  # type: ignore[assignment,misc]

try:
    from src.animation.retargeting import MediaPipeToSkeletonRetargeter
except ImportError:  # pragma: no cover
    MediaPipeToSkeletonRetargeter = None  # type: ignore[assignment,misc]

try:
    from src.animation.export import export_all
except ImportError:  # pragma: no cover
    def export_all(  # type: ignore[misc]
        skeleton: Any,
        output_dir: Path,
        formats: List[str],
        stem: str,
        **_kw: Any,
    ) -> Dict[str, Path]:
        logger.warning("animation.export not available; no files written")
        return {}

# evaluation
try:
    from src.evaluation.metrics import compute_metrics
except ImportError:  # pragma: no cover
    def compute_metrics(  # type: ignore[misc]
        seq: np.ndarray, **_kw: Any
    ) -> Optional[object]:
        return None

# tqdm (optional)
try:
    from tqdm import tqdm as _tqdm  # type: ignore[import-untyped]
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


def _progress(iterable, **kwargs):
    """Wrap *iterable* with tqdm if available, else return it unchanged."""
    if _TQDM_AVAILABLE:
        return _tqdm(iterable, **kwargs)
    return iterable


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProcessingResult:
    """Returned by :meth:`MocapPipeline.process`."""
    source_video: Path
    output_files: Dict[str, Path] = field(default_factory=dict)
    metrics: Optional[object] = None
    processing_time_sec: float = 0.0
    frame_count: int = 0
    detection_rate: float = 0.0  # fraction of frames with valid detection (0–1)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class MocapPipeline:
    """Orchestrates the full video → 3D pose → animation export pipeline.

    Parameters
    ----------
    settings:
        Fully-populated :class:`~src.config.settings.Settings` instance.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._log = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        video_path: Path,
        output_dir: Optional[Path] = None,
    ) -> ProcessingResult:
        """Full video → animation pipeline.

        Parameters
        ----------
        video_path:
            Path to the input video file.
        output_dir:
            Directory where output files are written.  Falls back to
            ``settings.animation.output_dir`` if *None*.

        Returns
        -------
        ProcessingResult
            Dataclass containing output paths, timing, and metrics.

        Raises
        ------
        FileNotFoundError
            If *video_path* does not exist.
        RuntimeError
            If the video has no readable frames.
        """
        pipeline_start = time.perf_counter()

        video_path = Path(video_path)
        if output_dir is None:
            output_dir = Path(self.settings.animation.output_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = video_path.stem

        # ---- 1. Validate inputs ----------------------------------------
        self._log.info("Stage 1/8: Validating inputs")
        t0 = time.perf_counter()
        validate_video_path(video_path)
        self._log.debug("  validation: %.3fs", time.perf_counter() - t0)

        # ---- 2. Extract video metadata ---------------------------------
        self._log.info("Stage 2/8: Extracting video metadata")
        t0 = time.perf_counter()
        meta = extract_metadata(video_path)
        self._log.info(
            "  %dx%d @ %.2f fps, %d frames, %.1fs",
            meta.width, meta.height, meta.fps, meta.frame_count, meta.duration_sec,
        )
        self._log.debug("  metadata: %.3fs", time.perf_counter() - t0)

        if meta.frame_count == 0:
            raise RuntimeError(f"Video appears to have no frames: {video_path}")

        # ---- 3. Initialize preprocessor --------------------------------
        self._log.info("Stage 3/8: Initializing processing components")
        t0 = time.perf_counter()
        preprocessor = FramePreprocessor(
            max_dimension=self.settings.video.max_dimension,
        )

        # ---- 4. Initialize MediaPipePose2D estimator -------------------
        pose2d = self._build_pose2d_estimator()

        # ---- 5. Initialize One Euro filter -----------------------------
        landmark_filter = self._build_landmark_filter(meta.fps)
        self._log.debug("  component init: %.3fs", time.perf_counter() - t0)

        # ---- 6. Per-frame estimation -----------------------------------
        self._log.info("Stage 4/8: Running per-frame pose estimation")
        t0 = time.perf_counter()

        world_seq: List[np.ndarray] = []   # [33, 3] per frame
        image_seq: List[np.ndarray] = []   # [33, 3] per frame (x, y, vis)
        empty_lm = np.full((33, 3), np.nan, dtype=np.float32)
        detected_count = 0

        with VideoLoader(video_path, target_fps=self.settings.video.target_fps) as loader:
            frame_iter = _progress(
                loader,
                desc="Estimating pose",
                unit="frame",
                total=meta.frame_count if meta.frame_count > 0 else None,
                dynamic_ncols=True,
            )
            for rgb_frame in frame_iter:
                proc_frame = preprocessor.process(rgb_frame)
                lm_world, lm_image = self._estimate_frame(pose2d, proc_frame)

                if lm_world is not None:
                    # Apply One Euro filter to world landmarks
                    if landmark_filter is not None:
                        lm_world = landmark_filter.update(lm_world)
                    world_seq.append(lm_world)
                    image_seq.append(lm_image if lm_image is not None else empty_lm.copy())
                    detected_count += 1
                else:
                    world_seq.append(empty_lm.copy())
                    image_seq.append(empty_lm.copy())

        frame_count = len(world_seq)
        self._log.info(
            "  %d frames collected, detection in %d (%.1f%%)",
            frame_count,
            detected_count,
            100.0 * detected_count / frame_count if frame_count else 0.0,
        )
        self._log.debug("  frame loop: %.3fs", time.perf_counter() - t0)

        if frame_count == 0:
            self._log.warning("No frames were read from the video — returning empty result.")
            return ProcessingResult(
                source_video=video_path,
                processing_time_sec=time.perf_counter() - pipeline_start,
                frame_count=0,
                detection_rate=0.0,
            )

        if detected_count == 0:
            self._log.warning(
                "No person was detected in any frame of %s — returning empty result.",
                video_path.name,
            )
            return ProcessingResult(
                source_video=video_path,
                processing_time_sec=time.perf_counter() - pipeline_start,
                frame_count=frame_count,
                detection_rate=0.0,
            )

        detection_rate = detected_count / frame_count

        # ---- 7a. Stack arrays [F, 33, 3] --------------------------------
        self._log.info("Stage 5/8: Post-processing pose sequence")
        t0 = time.perf_counter()
        world_arr = np.stack(world_seq, axis=0).astype(np.float32)   # (F, 33, 3)
        image_arr = np.stack(image_seq, axis=0).astype(np.float32)   # (F, 33, 3)

        # ---- 7b. Estimate body proportions -----------------------------
        body_props = estimate_body_proportions(
            world_arr,
            n_frames=self.settings.reconstruction.bone_estimation_frames,
        )
        self._log.debug("  body proportions: %s", body_props)

        # ---- 7c. PoseOptimizer: gap fill → bone normalize → SavGol ----
        optimizer = PoseOptimizer(
            settings=self.settings,
            body_proportions=body_props,
        )
        world_arr = optimizer.run(world_arr)
        self._log.debug("  optimization: %.3fs", time.perf_counter() - t0)

        # ---- 7d. Estimate camera model from first valid frame ----------
        self._log.info("Stage 6/8: Estimating camera model")
        t0 = time.perf_counter()
        first_valid = self._first_valid_frame(world_arr)
        camera = estimate_camera(
            world_pts=world_arr[first_valid] if first_valid is not None else world_arr[0],
            image_width=meta.width,
            image_height=meta.height,
            focal_length=self.settings.reconstruction.focal_length,
        )
        self._log.debug("  camera estimation: %.3fs", time.perf_counter() - t0)

        # ---- 7e. Build AnimationSkeleton --------------------------------
        self._log.info("Stage 7/8: Building skeleton and retargeting")
        t0 = time.perf_counter()
        skeleton = self._build_skeleton(world_arr, image_arr, meta, camera)

        # ---- 7f. Retarget ----------------------------------------------
        skeleton = self._retarget(skeleton, world_arr)
        self._log.debug("  skeleton + retarget: %.3fs", time.perf_counter() - t0)

        # ---- 7g. Export all formats ------------------------------------
        self._log.info("Stage 8/8: Exporting animation")
        t0 = time.perf_counter()
        output_files = export_all(
            skeleton=skeleton,
            output_dir=output_dir,
            formats=self.settings.animation.export_formats,
            stem=stem,
            fps=self.settings.video.target_fps,
            bvh_scale=self.settings.animation.bvh_scale,
            coordinate_system=self.settings.reconstruction.coordinate_system,
        )
        self._log.info("  exported: %s", list(output_files.keys()))
        self._log.debug("  export: %.3fs", time.perf_counter() - t0)

        # ---- 7h. Compute metrics ---------------------------------------
        metrics = compute_metrics(
            predicted=world_arr,
            image_pts=image_arr,
            fps=self.settings.video.target_fps,
            output_dir=output_dir,
            stem=stem,
            save_csv=self.settings.evaluation.save_metrics_csv,
            ground_truth_path=self.settings.evaluation.ground_truth_path,
        )

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
        """Construct the 2D pose estimator, respecting the configured backend."""
        if MediaPipePose2D is None:
            self._log.warning(
                "pose.pose_2d module not available — using built-in fallback estimator"
            )
            return _FallbackPose2D(self.settings)

        try:
            estimator = MediaPipePose2D(
                model_complexity=self.settings.pose.model_complexity,
                min_detection_confidence=self.settings.detection.min_detection_confidence,
                min_tracking_confidence=self.settings.detection.min_tracking_confidence,
                rotation_fallback_threshold=self.settings.detection.rotation_fallback_threshold,
                device=self.settings.device,
            )
            self._log.debug("MediaPipePose2D estimator initialized")
            return estimator
        except Exception as exc:
            self._log.warning(
                "Could not initialize MediaPipePose2D (%s); using fallback", exc
            )
            return _FallbackPose2D(self.settings)

    def _build_landmark_filter(self, source_fps: float) -> Any:
        """Construct the One Euro landmark filter."""
        if LandmarkFilter is None:
            self._log.debug("filtering.temporal_filter not available — skipping One Euro filter")
            return None

        try:
            filt = LandmarkFilter(
                n_landmarks=33,
                n_dims=3,
                min_cutoff=self.settings.filtering.one_euro_min_cutoff,
                beta=self.settings.filtering.one_euro_beta,
                d_cutoff=self.settings.filtering.one_euro_d_cutoff,
                fps=source_fps,
                mode="one_euro",
                realtime=True,
            )
            return filt
        except Exception as exc:
            self._log.warning("LandmarkFilter init failed (%s) — skipping filter", exc)
            return None

    def _estimate_frame(
        self,
        pose2d: Any,
        frame: np.ndarray,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Run pose estimation on a single frame.

        Returns
        -------
        (world_landmarks, image_landmarks) or (None, None) if no detection.
        Both arrays have shape (33, 3).  world_landmarks are metric coordinates;
        image_landmarks are (x_norm, y_norm, visibility).
        """
        try:
            result = pose2d.estimate(frame)
        except Exception as exc:
            self._log.debug("Pose estimation error: %s", exc)
            return None, None

        if result is None:
            return None, None

        # Expect result to expose .world_landmarks and .image_landmarks as (33,3) arrays.
        if hasattr(result, "world_landmarks") and hasattr(result, "image_landmarks"):
            return result.world_landmarks, result.image_landmarks

        # Legacy / fallback: result is a 2-tuple (world, image)
        if isinstance(result, (tuple, list)) and len(result) == 2:
            return result[0], result[1]

        self._log.warning("Unexpected pose result type: %s", type(result))
        return None, None

    def _build_skeleton(
        self,
        world_arr: np.ndarray,
        image_arr: np.ndarray,
        meta: Any,
        camera: Any,
    ) -> Any:
        """Instantiate AnimationSkeleton with the processed sequence."""
        if AnimationSkeleton is None:
            return _StubSkeleton(world_arr, image_arr, meta)

        try:
            return AnimationSkeleton(
                world_sequence=world_arr,
                image_sequence=image_arr,
                fps=self.settings.video.target_fps,
                camera=camera,
                coordinate_system=self.settings.reconstruction.coordinate_system,
            )
        except Exception as exc:
            self._log.warning("AnimationSkeleton init failed (%s) — using stub", exc)
            return _StubSkeleton(world_arr, image_arr, meta)

    def _retarget(self, skeleton: Any, world_arr: np.ndarray) -> Any:
        """Apply MediaPipe → skeleton retargeting if available."""
        if MediaPipeToSkeletonRetargeter is None:
            self._log.debug("animation.retargeting not available — skipping retarget")
            return skeleton

        try:
            retargeter = MediaPipeToSkeletonRetargeter(
                coordinate_system=self.settings.reconstruction.coordinate_system,
            )
            return retargeter.retarget(skeleton, world_arr)
        except Exception as exc:
            self._log.warning("Retargeting failed (%s) — using un-retargeted skeleton", exc)
            return skeleton

    @staticmethod
    def _first_valid_frame(arr: np.ndarray) -> Optional[int]:
        """Return the index of the first frame with no NaN landmarks."""
        for i, frame in enumerate(arr):
            if not np.any(np.isnan(frame)):
                return i
        return None


# ---------------------------------------------------------------------------
# Internal stubs (used when real modules are absent)
# ---------------------------------------------------------------------------

class _FallbackPose2D:
    """Minimal MediaPipe-based pose estimator used when pose_2d module is absent."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._landmarker = None
        self._prev_rot = None
        self._rotations = None
        self._init_landmarker()

    def _init_landmarker(self) -> None:
        import os
        import urllib.request

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
        except ImportError:
            logger.error("mediapipe not installed — pose estimation unavailable")
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

    def estimate(self, frame: np.ndarray):
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
            # Unrotate coordinates back to original frame orientation
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

        # Try previous rotation first
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
        # Zero out low-visibility landmarks
        low_vis = img_pts[:, 2] < min_vis
        world_pts[low_vis] = np.nan
        return world_pts, img_pts


@dataclass
class _StubSkeleton:
    """Passthrough skeleton used when animation.skeleton is unavailable."""
    world_sequence: np.ndarray
    image_sequence: np.ndarray
    meta: Any
