"""
Video -> smoothed 3D pose data (JSON) at 30 fps, for blender_apply.py.

Setup (once):
    pip install mediapipe opencv-python scipy numpy

Run:
    python extract_pose.py pole_2.mp4 --preview
    -> pole_2_pose.json  (+ pole_2_preview.mp4 with the skeleton drawn on top)

Options:
    --fps 30        sample rate of the output animation
    --cutoff 6      smoothing strength in Hz (lower = smoother, higher = snappier; try 4-10)
    --no-rotate     turn off the upside-down trick (faster, worse on flips/inversions)
"""
import argparse
import json
import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from scipy.signal import butter, filtfilt

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_heavy.task")

NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]
BODY = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # main joints, used to score a detection
SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32)]
ROTATIONS = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
MIN_VISIBILITY = 0.3


def load_landmarker():
    if not os.path.exists(MODEL_PATH):
        print("Downloading pose model (one time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
    )
    return vision.PoseLandmarker.create_from_options(options)


def unrotate(rot, u, v, centered):
    """Map coordinates from a rotated image back to the original image orientation.
    centered=True for hip-centered world coords, False for 0..1 image coords."""
    if rot is None:
        return u, v
    if rot == cv2.ROTATE_180:
        return (-u, -v) if centered else (1 - u, 1 - v)
    if rot == cv2.ROTATE_90_CLOCKWISE:
        return (v, -u) if centered else (v, 1 - u)
    return (-v, u) if centered else (1 - v, u)  # counter-clockwise


def detect(landmarker, rgb, rot):
    """Run pose detection on the frame rotated by `rot`. Returns (score, image_pts, world_pts) or None."""
    img = rgb if rot is None else cv2.rotate(rgb, rot)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
    if not result.pose_landmarks:
        return None
    image_pts = np.array([[p.x, p.y, p.visibility] for p in result.pose_landmarks[0]])
    world_pts = np.array([[p.x, p.y, p.z] for p in result.pose_world_landmarks[0]])
    image_pts[:, 0], image_pts[:, 1] = unrotate(rot, image_pts[:, 0], image_pts[:, 1], centered=False)
    world_pts[:, 0], world_pts[:, 1] = unrotate(rot, world_pts[:, 0], world_pts[:, 1], centered=True)
    score = image_pts[BODY, 2].mean()
    return score, image_pts, world_pts


def best_detection(landmarker, rgb, prev_rot, try_rotations):
    """Detect the pose; for inverted/sideways bodies also try the frame rotated 90/180/270 degrees."""
    results = {}
    results[prev_rot] = detect(landmarker, rgb, prev_rot)
    first = results[prev_rot]
    if try_rotations and (first is None or first[0] < 0.8):
        for rot in ROTATIONS:
            if rot not in results:
                results[rot] = detect(landmarker, rgb, rot)
    best_rot, best = prev_rot, first
    for rot, res in results.items():
        if res is None:
            continue
        # stick with the previous rotation unless another one is clearly better (prevents flicker)
        if best is None or res[0] > best[0] + 0.1:
            best_rot, best = rot, res
    return best_rot, best


def fill_gaps(arr):
    """Linearly interpolate NaNs along time for every column of a (frames, ...) array."""
    flat = arr.reshape(arr.shape[0], -1)
    t = np.arange(flat.shape[0])
    for c in range(flat.shape[1]):
        col = flat[:, c]
        good = ~np.isnan(col)
        if good.sum() == 0:
            col[:] = 0.0
        elif good.sum() < len(col):
            col[~good] = np.interp(t[~good], t[good], col[good])
    return flat.reshape(arr.shape)


def smooth(arr, fps, cutoff):
    """Zero-lag Butterworth low-pass along time. Removes jitter without delaying the motion."""
    if arr.shape[0] < 15 or cutoff <= 0 or cutoff >= fps / 2:
        return arr
    b, a = butter(2, cutoff / (fps / 2))
    return filtfilt(b, a, arr, axis=0)


def postprocess(image_pts, world_pts, width, height, fps, cutoff):
    """image_pts: (F,33,3) x,y,visibility in 0..1 (NaN if no detection). world_pts: (F,33,3) meters."""
    vis = np.nan_to_num(image_pts[:, :, 2], nan=0.0)
    low = vis < MIN_VISIBILITY
    world = world_pts.copy()
    px = image_pts[:, :, 0] * width
    py = image_pts[:, :, 1] * height
    world[low] = np.nan
    px[low] = np.nan
    py[low] = np.nan
    world = fill_gaps(world)
    px = fill_gaps(px)
    py = fill_gaps(py)

    # keep the body centered on the hip midpoint
    hip_mid = (world[:, 23] + world[:, 24]) / 2
    world = world - hip_mid[:, None, :]

    # rough root motion: how far the hips move in the image, converted to meters
    hip_px = (px[:, 23] + px[:, 24]) / 2
    hip_py = (py[:, 23] + py[:, 24]) / 2
    sh_px = (px[:, 11] + px[:, 12]) / 2
    sh_py = (py[:, 11] + py[:, 12]) / 2
    torso_pixels = np.hypot(sh_px - hip_px, sh_py - hip_py)
    torso_meters = np.linalg.norm((world[:, 11] + world[:, 12]) / 2, axis=1)
    meters_per_pixel = np.median(torso_meters / np.maximum(torso_pixels, 1e-6))
    root = np.zeros((world.shape[0], 3))
    root[:, 0] = (hip_px - hip_px[0]) * meters_per_pixel
    root[:, 1] = (hip_py - hip_py[0]) * meters_per_pixel

    world = smooth(world, fps, cutoff)
    root = smooth(root, fps, cutoff)

    # MediaPipe axes (x right, y down, z away from camera) -> Blender axes (X right, Y away, Z up)
    def to_blender(a):
        return np.stack([a[..., 0], a[..., 2], -a[..., 1]], axis=-1)

    world = to_blender(world)
    root = to_blender(root)

    legs = []
    for hip, knee, ankle in [(23, 25, 27), (24, 26, 28)]:
        legs.append(np.linalg.norm(world[:, hip] - world[:, knee], axis=1) +
                    np.linalg.norm(world[:, knee] - world[:, ankle], axis=1))
    leg_length = float(np.median(np.mean(legs, axis=0)))
    return world, root, leg_length


def draw(frame, image_pts):
    h, w = frame.shape[:2]
    for a, b in SKELETON:
        if image_pts[a, 2] > MIN_VISIBILITY and image_pts[b, 2] > MIN_VISIBILITY:
            pa = (int(image_pts[a, 0] * w), int(image_pts[a, 1] * h))
            pb = (int(image_pts[b, 0] * w), int(image_pts[b, 1] * h))
            cv2.line(frame, pa, pb, (0, 255, 0), 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--no-rotate", action="store_true")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    base = os.path.splitext(args.video)[0]
    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if args.preview:
        writer = cv2.VideoWriter(base + "_preview.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.fps, (width, height))

    landmarker = load_landmarker()
    image_list, world_list = [], []
    rot = None
    frame_index = 0
    next_time = 0.0
    empty = np.full((33, 3), np.nan)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_index / src_fps
        frame_index += 1
        if t + 1e-6 < next_time:
            continue  # skip frames between 30 fps samples
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rot, res = best_detection(landmarker, rgb, rot, not args.no_rotate)
        # repeat the sample if the source fps is lower than the target fps
        while next_time <= t + 1e-6:
            if res is None:
                image_list.append(empty)
                world_list.append(empty)
            else:
                image_list.append(res[1])
                world_list.append(res[2])
            next_time += 1.0 / args.fps
        if writer is not None:
            if res is not None:
                draw(frame, res[1])
            writer.write(frame)
        n = len(image_list)
        if n % 30 == 0:
            print(f"  {n} frames processed")

    cap.release()
    if writer is not None:
        writer.release()
    if not image_list:
        raise SystemExit("No frames read from video.")

    image_pts = np.array(image_list)
    world_pts = np.array(world_list)
    detected = int((~np.isnan(image_pts[:, 0, 0])).sum())
    print(f"Detected a pose in {detected}/{len(image_pts)} frames")

    world, root, leg_length = postprocess(image_pts, world_pts, width, height, args.fps, args.cutoff)
    out = {
        "fps": args.fps,
        "names": NAMES,
        "leg_length": leg_length,
        "root": np.round(root, 5).tolist(),
        "frames": np.round(world, 5).tolist(),
    }
    with open(base + "_pose.json", "w") as f:
        json.dump(out, f)
    print("Wrote", base + "_pose.json")


if __name__ == "__main__":
    main()