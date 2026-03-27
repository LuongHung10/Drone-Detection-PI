"""
tracker_utils.py - Tracking helper utilities

Includes:
  - Kalman Filter creation / update
  - IoU calculation
  - Hungarian matching
  - Adaptive confidence adjustment
  - Adaptive image-size adjustment
  - Occlusion detection
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

import config as cfg

# =====================================================
# OPTIONAL DEPENDENCY: filterpy (Kalman Filter)
# =====================================================
try:
    from filterpy.kalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    print(
        "Warning: filterpy not installed. Kalman Filter disabled. "
        "Install with: pip install filterpy"
    )

# =====================================================
# MODULE-LEVEL STATE (initialised / reset by caller)
# =====================================================
current_adaptive_conf: float = cfg.CONF_THRESH
detection_quality_history: list[float] = []
adaptive_imgsz_history: list[int] = []
scene_complexity: float = 0.5


def reset_state():
    """Reset all module-level mutable state (call when starting a new video)."""
    global current_adaptive_conf, detection_quality_history
    global adaptive_imgsz_history, scene_complexity
    current_adaptive_conf = cfg.CONF_THRESH
    detection_quality_history.clear()
    adaptive_imgsz_history.clear()
    scene_complexity = 0.5


# =====================================================
# KALMAN FILTER
# =====================================================
# def create_kalman_filter(x: float, y: float, w: float, h: float):
#     """Create a Kalman Filter for a new track. Returns None if filterpy missing."""
#     if not KALMAN_AVAILABLE or not cfg.USE_KALMAN_FILTER:
#         return None

#     kf = KalmanFilter(dim_x=8, dim_z=4)

#     # State: [x, y, w, h, vx, vy, vw, vh]
#     kf.x = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)

#     # Constant-velocity transition
#     kf.F = np.array([
#         [1, 0, 0, 0, 1, 0, 0, 0],
#         [0, 1, 0, 0, 0, 1, 0, 0],
#         [0, 0, 1, 0, 0, 0, 1, 0],
#         [0, 0, 0, 1, 0, 0, 0, 1],
#         [0, 0, 0, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 0, 0, 1, 0],
#         [0, 0, 0, 0, 0, 0, 0, 1],
#     ], dtype=np.float32)

#     # Measurement: observe x, y, w, h only
#     kf.H = np.zeros((4, 8), dtype=np.float32)
#     for i in range(4):
#         kf.H[i, i] = 1.0

#     kf.P *= 1000.0
#     kf.R = np.eye(4, dtype=np.float32) * cfg.KALMAN_MEASUREMENT_NOISE
#     kf.Q = np.eye(8, dtype=np.float32) * cfg.KALMAN_PROCESS_NOISE

#     return kf


# def update_kalman_filter(kf, x: float, y: float, w: float, h: float) -> np.ndarray:
#     """
#     Predict + update a Kalman Filter with a new measurement.
#     Returns the smoothed [x, y, w, h].
#     """
#     if kf is None or not KALMAN_AVAILABLE:
#         return np.array([x, y, w, h], dtype=np.float32)
#     kf.predict()
#     kf.update(np.array([x, y, w, h], dtype=np.float32))
#     return kf.x[:4]


# =====================================================
# IoU
# =====================================================
def calculate_iou(box1, box2) -> float:
    """Return Intersection-over-Union of two [x1, y1, x2, y2] boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    ix1 = max(x1_min, x2_min)
    iy1 = max(y1_min, y2_min)
    ix2 = min(x1_max, x2_max)
    iy2 = min(y1_max, y2_max)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# =====================================================
# HUNGARIAN MATCHING
# =====================================================
def hungarian_matching(tracks, detections, iou_thresh: float = 0.5) -> list[tuple]:
    """
    Match existing tracks to new detections using the Hungarian algorithm.

    Args:
        tracks:     list of [x1, y1, x2, y2] for current tracks
        detections: list of [x1, y1, x2, y2] for new detections
        iou_thresh: minimum IoU to accept a match

    Returns:
        list of (track_idx, detection_idx, iou) tuples
    """
    if not tracks or not detections:
        return []

    cost = np.array(
        [[1.0 - calculate_iou(t, d) for d in detections] for t in tracks]
    )
    row_idx, col_idx = linear_sum_assignment(cost)

    return [
        (r, c, 1.0 - cost[r, c])
        for r, c in zip(row_idx, col_idx)
        if (1.0 - cost[r, c]) >= iou_thresh
    ]


# =====================================================
# ADAPTIVE CONFIDENCE
# =====================================================
def adaptive_confidence_adjustment(quality_history: list[float]) -> float:
    """
    Adjust the confidence threshold based on recent detection quality.
    A low detection rate drives the threshold down; high quality drives it up.
    """
    global current_adaptive_conf

    if not cfg.USE_ADAPTIVE_CONF or len(quality_history) < 2:
        return current_adaptive_conf

    recent = quality_history[-8:]
    avg_quality = float(np.mean(recent))
    det_rate = sum(1 for q in recent if q > 0) / len(recent)

    if det_rate < 0.60:
        current_adaptive_conf = max(
            cfg.ADAPTIVE_CONF_MIN,
            current_adaptive_conf - cfg.ADAPTIVE_CONF_STEP * 2.5,
        )
    elif det_rate < 0.70:
        current_adaptive_conf = max(
            cfg.ADAPTIVE_CONF_MIN,
            current_adaptive_conf - cfg.ADAPTIVE_CONF_STEP,
        )
    elif avg_quality < 0.5:
        current_adaptive_conf = max(
            cfg.ADAPTIVE_CONF_MIN,
            current_adaptive_conf - cfg.ADAPTIVE_CONF_STEP * 0.5,
        )
    elif avg_quality > 0.85:
        current_adaptive_conf = min(
            cfg.ADAPTIVE_CONF_MAX,
            current_adaptive_conf + cfg.ADAPTIVE_CONF_STEP,
        )

    return current_adaptive_conf


# =====================================================
# ADAPTIVE IMAGE SIZE
# =====================================================
def adaptive_imgsz_adjustment(detection_count: int, last_detection_count: int) -> int:
    """Return the best inference image size based on scene complexity."""
    global scene_complexity

    if not cfg.USE_ADAPTIVE_IMGSZ:
        return cfg.IMGSZ

    if detection_count == 0 and last_detection_count == 0:
        scene_complexity = max(0.0, scene_complexity - 0.05)
    elif detection_count > 0:
        scene_complexity = min(1.0, scene_complexity + 0.02)

    if scene_complexity < 0.3:
        imgsz = cfg.ADAPTIVE_IMGSZ_MIN
    elif scene_complexity > 0.7:
        imgsz = cfg.ADAPTIVE_IMGSZ_MAX
    else:
        imgsz = cfg.ADAPTIVE_IMGSZ_BASE

    adaptive_imgsz_history.append(imgsz)
    if len(adaptive_imgsz_history) > 10:
        adaptive_imgsz_history.pop(0)

    return imgsz


# =====================================================
# OCCLUSION DETECTION
# =====================================================
def detect_occlusion(boxes, iou_thresh: float = 0.3) -> set[int]:
    """
    Return indices of boxes that are occluded by a larger overlapping box.
    """
    occluded: set[int] = set()
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if calculate_iou(boxes[i], boxes[j]) > iou_thresh:
                area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
                area_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
                occluded.add(i if area_i < area_j else j)
    return occluded