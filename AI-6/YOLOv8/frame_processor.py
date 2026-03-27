"""
frame_processor.py - Per-frame inference, tracking, and annotation logic
"""
import time
from collections import defaultdict

import cv2
import numpy as np

import config as cfg
import perf_monitor
import utils as tu
from targeting_overlay import draw_targeting_overlay, reset_lock, is_locked, get_locked_tid
from utils import (
    KALMAN_AVAILABLE,
    adaptive_confidence_adjustment,
    # create_kalman_filter,
    detect_occlusion,
    # update_kalman_filter,
)

# =====================================================
# MUTABLE TRACKING STATE  (reset via reset_state())
# =====================================================
tracking_history: dict = {}
# kalman_filters: dict = {}
occluded_tracks: dict = {}
last_frame_results: dict | None = None
last_annotated_frame = None 
skip_frame_counter: int = 0
tracking_stability: dict = {}


def reset_state():
    """Clear all per-video tracking state. Call before processing each new video."""
    global tracking_history #, kalman_filters
    global occluded_tracks, last_frame_results, skip_frame_counter, tracking_stability
    global last_annotated_frame

    tracking_history.clear()
    # kalman_filters.clear()
    occluded_tracks.clear()
    last_frame_results = None
    last_annotated_frame = None
    skip_frame_counter = 0
    tracking_stability.clear()

    tu.reset_state()
    perf_monitor.reset()
    reset_lock()


# =====================================================
# MAIN FRAME PROCESSING
# =====================================================
def process_frame(model, device: str, frame, frame_count: int = 0,
                  total_frames: int = 0, fps: float = 0.0,
                  logger=None) -> tuple:
    """
    Run inference + tracking + annotation on a single BGR frame.

    Returns:
        (annotated_frame, stats_dict)
    """
    global tracking_history #, kalman_filters
    global occluded_tracks, last_frame_results, skip_frame_counter, tracking_stability

    frame_start = time.time()

    # Skip frames when locked to save CPU (but still run tracking updates internally)
    if (cfg.USE_FRAME_SKIP
            and not is_locked()
            and frame_count % cfg.FRAME_SKIP_N != 0
            and last_annotated_frame is not None):
        # ByteTrack continues to predict positions internally between detections
        return last_annotated_frame, {
            'detection_count': 0,
            'track_count': 0,
            'class_counts': {},
            'track_ids': [],
            'inference_time_ms': 0,
            'total_time_ms': (time.time() - frame_start) * 1000,
        }

    # ── Adaptive confidence ──────────────────────────────────────────────────
    conf = cfg.CONF_THRESH
    if cfg.USE_ADAPTIVE_CONF:
        conf = adaptive_confidence_adjustment(tu.detection_quality_history)
        if cfg.ENABLE_TRACKING:
            conf = max(conf, cfg.TRACK_CONF_THRESH)

    # ── Adaptive image size ──────────────────────────────────────────────────
    imgsz = cfg.IMGSZ
    if cfg.USE_ADAPTIVE_IMGSZ and len(tu.detection_quality_history) > 3:
        recent_no_det = sum(
            1 for q in tu.detection_quality_history[-8:] if q == 0
        )
        det_rate = 1.0 - (recent_no_det / min(8, len(tu.detection_quality_history)))
        if det_rate < 0.50:
            imgsz = cfg.ADAPTIVE_IMGSZ_MAX
        elif det_rate > 0.75:
            imgsz = cfg.ADAPTIVE_IMGSZ_MIN
        else:
            imgsz = cfg.ADAPTIVE_IMGSZ_BASE

    # ── Smart multi-scale boost ──────────────────────────────────────────────
    if (cfg.USE_SMART_MULTI_SCALE
            and tu.detection_quality_history
            and frame_count % cfg.MULTI_SCALE_FRAME_INTERVAL == 0):
        recent_dets = sum(
            1 for q in tu.detection_quality_history[-5:] if q > 0
        )
        if recent_dets <= cfg.MULTI_SCALE_THRESHOLD:
            imgsz = int(imgsz * 1.15)

    # ── Max detections cap (Pi 5 memory saving) ─────────────────────────────
    from hardware import IS_RASPBERRY_PI5, HAS_HAILO_NPU
    if IS_RASPBERRY_PI5 and not HAS_HAILO_NPU:
        max_det = 150
    elif IS_RASPBERRY_PI5 and HAS_HAILO_NPU:
        max_det = 250
    else:
        max_det = 500

    # ── Inference ────────────────────────────────────────────────────────────
    inf_start = time.time()

    # if cfg.ENABLE_TRACKING and cfg.TRACKER_TYPE in ("bytetrack", "botsort", "hybrid"):
    #     tracker_yaml = (
    #         "botsort.yaml" if cfg.TRACKER_TYPE == "botsort" else "bytetrack.yaml"
    #     )
    #     results = model.track(
    #         frame,
    #         conf=conf,
    #         iou=cfg.TRACK_IOU_THRESH,
    #         imgsz=imgsz,
    #         verbose=cfg.DEBUG_DETECTIONS,
    #         device=device,
    #         half=cfg.USE_HALF,
    #         persist=True,
    #         tracker=tracker_yaml,
    #         show=False,
    #         agnostic_nms=False,
    #         max_det=max_det,
    #         stream=False,
    #     )
    # else:
    #     results = model.predict(
    #         frame,
    #         conf=conf,
    #         iou=cfg.IOU_THRESH,
    #         imgsz=imgsz,
    #         verbose=cfg.DEBUG_DETECTIONS,
    #         device=device,
    #         half=cfg.USE_HALF,
    #         show=False,
    #         agnostic_nms=False,
    #         max_det=max_det,
    #         stream=False,
    #     )

    def _predict_fallback():
        """Plain predict with no tracker."""
        return model.predict(
            frame,
            conf=conf,
            iou=cfg.IOU_THRESH,
            imgsz=imgsz,
            verbose=cfg.DEBUG_DETECTIONS,
            device=device,
            half=cfg.USE_HALF,
            show=False,
            agnostic_nms=False,
            max_det=max_det,
            retina_masks=False,
            stream=False,
        )
 
    if cfg.ENABLE_TRACKING and cfg.TRACKER_TYPE in ("bytetrack", "botsort", "hybrid"):
        tracker_yaml = (
            "botsort.yaml" if cfg.TRACKER_TYPE == "botsort" else "bytetrack.yaml"
        )
        try:
            results = model.track(
                frame,
                conf=conf,
                iou=cfg.TRACK_IOU_THRESH,
                imgsz=imgsz,
                verbose=cfg.DEBUG_DETECTIONS,
                device=device,
                half=cfg.USE_HALF,
                persist=True,
                tracker=tracker_yaml,
                show=False,
                agnostic_nms=False,
                max_det=max_det,
                retina_masks=False,
                stream=False,
            )
        except Exception as e:
            # ByteTrack Kalman filter can go numerically unstable with NCNN
            # Reset the tracker and fall back to plain predict for this frame
            if logger:
                logger.warning(f"Tracker error (resetting): {e}")
            try:
                model.predictor.trackers[0].reset()
            except Exception:
                pass
            results = _predict_fallback()
    else:
        results = _predict_fallback()

    inference_time = time.time() - inf_start
    post_start = time.time()

    names = model.names
    raw_det_count = sum(len(r.boxes) for r in results if r.boxes is not None)

    # ── Filter to locked target only when locked ─────────────────────────────
    locked_tid = get_locked_tid() if is_locked() else None

    annotated = frame.copy()
    detection_count = 0
    track_count = 0
    class_counts: dict[str, int] = defaultdict(int)
    track_ids_set: set[int] = set()

    # ── Per-result annotation ────────────────────────────────────────────────
    for r in results:
        track_ids = None
        if cfg.ENABLE_TRACKING and r.boxes is not None and r.boxes.id is not None:
            track_ids = r.boxes.id.cpu().numpy().astype(int)
            track_count = len(track_ids)

        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes_data = r.boxes.xyxy.cpu().numpy()
        confs_arr = r.boxes.conf.cpu().numpy()
        clss_arr = r.boxes.cls.cpu().numpy().astype(int)

        # KCF hybrid update
        # if cfg.TRACKER_TYPE == "hybrid" and track_ids is not None:
        #     _update_kcf_trackers(frame, boxes_data, track_ids)

        # Occlusion detection
        occluded_idx = (
            detect_occlusion(boxes_data, cfg.OCCLUSION_THRESH)
            if cfg.USE_OCCLUSION_HANDLING and len(boxes_data) > 1
            else set()
        )

        for i, (box, conf_val, cls) in enumerate(zip(boxes_data, confs_arr, clss_arr)):
            x1, y1, x2, y2 = map(int, box)
            label = names[cls]
            w, h = x2 - x1, y2 - y1

            tid = track_ids[i] if (track_ids is not None and i < len(track_ids)) else None

            # ── Skip non-locked tracks when locked ────────────────────────────
            if locked_tid is not None and tid != locked_tid:
                continue

            # Kalman smoothing
            # if cfg.USE_KALMAN_FILTER and KALMAN_AVAILABLE and tid is not None:
            #     x1, y1, x2, y2 = _apply_kalman(tid, x1, y1, x2, y2, w, h, logger)
            #     w, h = x2 - x1, y2 - y1

            # Temporal smoothing
            if cfg.USE_TEMPORAL_SMOOTHING and tid is not None:
                x1, y1, x2, y2 = _apply_temporal_smoothing(tid, x1, y1, x2, y2)

            # Occlusion skip
            if cfg.USE_OCCLUSION_HANDLING and tid is not None:
                if i in occluded_idx:
                    occluded_tracks[tid] = occluded_tracks.get(tid, 0) + 1
                    if occluded_tracks[tid] > cfg.OCCLUSION_BUFFER:
                        continue
                else:
                    occluded_tracks[tid] = 0

            # KCF weighted merge
            # if cfg.TRACKER_TYPE == "hybrid" and tid is not None and tid in kcf_boxes:
            #     x1, y1, x2, y2 = _merge_kcf_box(tid, x1, y1, x2, y2)

            # Colour from track ID
            color = _track_color(tid)

            # Draw box
            thickness = 2 if (
                cfg.USE_OCCLUSION_HANDLING and tid and occluded_tracks.get(tid, 0) > 0
            ) else (3 if tid is not None else 2)

            if cfg.USE_OCCLUSION_HANDLING and tid and occluded_tracks.get(tid, 0) > 0:
                color = tuple(int(c * 0.5) for c in color)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Label
            tracker_tag = "KCF" if cfg.TRACKER_TYPE == "hybrid" else "BT"
            lbl = (
                f"ID:{tid}[{tracker_tag}] {label} {conf_val:.2f}"
                if tid is not None
                else f"{label} {conf_val:.2f}"
            )
            _draw_label(annotated, lbl, x1, y1, color)

            detection_count += 1
            class_counts[label] += 1
            if tid is not None:
                track_ids_set.add(tid)

    # ── HUD overlay ─────────────────────────────────────────────────────────
    _draw_hud(annotated, frame_count, total_frames, detection_count,
              raw_det_count, track_count, fps, tu.current_adaptive_conf,
              device)

    # ── Logging ──────────────────────────────────────────────────────────────
    if (logger and cfg.LOG_EVERY_N_FRAMES > 0
            and frame_count % cfg.LOG_EVERY_N_FRAMES == 0):
        classes_str = ", ".join(f"{k}:{v}" for k, v in class_counts.items())
        logger.info(
            f"Frame {frame_count}/{total_frames}: Det={detection_count}, "
            f"Tracks={track_count}, FPS={fps:.1f}"
            + (f" | Classes: {classes_str}" if classes_str else "")
        )

    if cfg.DEBUG_DETECTIONS and detection_count == 0 and frame_count % 30 == 0:
        msg = f"⚠️ Frame {frame_count}: No detections (Conf:{cfg.CONF_THRESH}, ImgSz:{cfg.IMGSZ})"
        print(msg)
        if logger:
            logger.warning(msg)

    # ── Adaptive conf quality update ─────────────────────────────────────────
    if cfg.USE_ADAPTIVE_CONF:
        if raw_det_count > 0 and detection_count == 0:
            quality = 0.1
        elif raw_det_count > detection_count * 1.5:
            quality = 0.3
        else:
            quality = 1.0 if detection_count > 0 else 0.0

        tu.detection_quality_history.append(quality)
        if len(tu.detection_quality_history) > 30:
            tu.detection_quality_history.pop(0)

    # ── Stale Kalman filter cleanup ───────────────────────────────────────────
    # if cfg.USE_KALMAN_FILTER and track_ids_set:
    #     for tid in list(kalman_filters.keys()):
    #         if tid not in track_ids_set:
    #             del kalman_filters[tid]

    # ── Performance stats ─────────────────────────────────────────────────────
    postprocess_time = time.time() - post_start
    total_time = time.time() - frame_start
    perf_monitor.update(inference_time, 0, postprocess_time)

    # ── Memory cleanup (Pi 5) ─────────────────────────────────────────────────
    if (cfg.ENABLE_MEMORY_OPTIMIZATION
            and frame_count > 0
            and frame_count % cfg.MEMORY_CLEANUP_INTERVAL == 0):
        perf_monitor.cleanup_memory()

    # ── Targeting overlay ────────────────────────────────────────────────────
    # Build a list of dicts sorted by confidence (highest first)
    targeting_dets = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        tids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [None] * len(r.boxes)
        for box, conf_val, cls, t in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int),
            tids,
        ):
            targeting_dets.append({
                'box':   tuple(map(int, box)),
                'label': names[cls],
                'conf':  float(conf_val),
                'tid':   int(t) if t is not None else None,
            })
    targeting_dets.sort(key=lambda d: d['conf'], reverse=True)
    draw_targeting_overlay(annotated, targeting_dets)

    stats = {
        'detection_count': detection_count,
        'track_count': track_count,
        'class_counts': dict(class_counts),
        'track_ids': list(track_ids_set),
        'inference_time_ms': inference_time * 1000,
        'total_time_ms': total_time * 1000,
    }
    return annotated, stats


# =====================================================
# PRIVATE HELPERS
# =====================================================
# def _apply_kalman(tid, x1, y1, x2, y2, w, h, logger):
#     if tid not in kalman_filters:
#         kf = create_kalman_filter(x1 + w / 2, y1 + h / 2, w, h)
#         if kf is not None:
#             kalman_filters[tid] = kf
#         return x1, y1, x2, y2
#     try:
#         kf = kalman_filters[tid]
#         pred = update_kalman_filter(kf, x1 + w / 2, y1 + h / 2, w, h)
#         kw = 0.3
#         x1 = int((1 - kw) * x1 + kw * (pred[0] - pred[2] / 2))
#         y1 = int((1 - kw) * y1 + kw * (pred[1] - pred[3] / 2))
#         x2 = int((1 - kw) * x2 + kw * (pred[0] + pred[2] / 2))
#         y2 = int((1 - kw) * y2 + kw * (pred[1] + pred[3] / 2))
#     except Exception as e:
#         if logger:
#             logger.debug(f"Kalman error: {e}")
#     return x1, y1, x2, y2


def _apply_temporal_smoothing(tid, x1, y1, x2, y2):
    history = tracking_history.setdefault(tid, [])
    history.append((x1, y1, x2, y2))
    if len(history) > cfg.SMOOTHING_HISTORY:
        history.pop(0)
    if len(history) > 1:
        weights = np.exp(np.linspace(-1, 0, len(history)))
        weights /= weights.sum()
        arr = np.array(history)
        x1 = int(np.average(arr[:, 0], weights=weights))
        y1 = int(np.average(arr[:, 1], weights=weights))
        x2 = int(np.average(arr[:, 2], weights=weights))
        y2 = int(np.average(arr[:, 3], weights=weights))
    return x1, y1, x2, y2


# def _update_kcf_trackers(frame, boxes_data, track_ids):
#     active = set(track_ids)
#     for tid in list(kcf_trackers.keys()):
#         if tid not in active:
#             kcf_trackers.pop(tid, None)
#             kcf_boxes.pop(tid, None)

#     for box, tid in zip(boxes_data, track_ids):
#         x1, y1, x2, y2 = map(int, box)
#         w, h = x2 - x1, y2 - y1
#         if tid not in kcf_trackers:
#             tracker = cv2.TrackerKCF_create()
#             tracker.init(frame, (x1, y1, w, h))
#             kcf_trackers[tid] = tracker
#             kcf_boxes[tid] = (x1, y1, x2, y2)
#         else:
#             try:
#                 ok, bbox = kcf_trackers[tid].update(frame)
#                 if ok:
#                     kx1, ky1, kw, kh = map(int, bbox)
#                     alpha = 0.7
#                     kcf_boxes[tid] = (
#                         int(alpha * x1 + (1 - alpha) * kx1),
#                         int(alpha * y1 + (1 - alpha) * ky1),
#                         int(alpha * x2 + (1 - alpha) * (kx1 + kw)),
#                         int(alpha * y2 + (1 - alpha) * (ky1 + kh)),
#                     )
#                 else:
#                     kcf_boxes[tid] = (x1, y1, x2, y2)
#             except Exception:
#                 kcf_boxes[tid] = (x1, y1, x2, y2)


# def _merge_kcf_box(tid, x1, y1, x2, y2):
#     kx1, ky1, kx2, ky2 = kcf_boxes[tid]
#     a = cfg.SMOOTHING_ALPHA
#     return (
#         int(a * x1 + (1 - a) * kx1),
#         int(a * y1 + (1 - a) * ky1),
#         int(a * x2 + (1 - a) * kx2),
#         int(a * y2 + (1 - a) * ky2),
#     )


def _track_color(tid) -> tuple:
    if tid is None:
        return (0, 255, 0)
    c = tid % 255
    return (
        int(255 * np.sin(c * 0.1) ** 2),
        int(255 * np.sin(c * 0.1 + 2) ** 2),
        int(255 * np.sin(c * 0.1 + 4) ** 2),
    )


def _draw_label(img, text: str, x1: int, y1: int, color: tuple):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, max(y1 - 5, th)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def _draw_hud(img, frame_count, total_frames, det_count, raw_count,
              track_count, fps, adaptive_conf, device):
    import torch as _torch
    tracker_tag = cfg.TRACKER_TYPE.upper() if cfg.ENABLE_TRACKING else "NONE"
    device_tag = "GPU" if _torch.cuda.is_available() else "CPU"

    if total_frames > 0:
        info = f"Frame: {frame_count}/{total_frames} | Det: {det_count}"
    else:
        info = f"Frame: Live | Det: {det_count}"

    if cfg.DEBUG_DETECTIONS and raw_count > det_count:
        info += f" (Raw: {raw_count})"
    if cfg.ENABLE_TRACKING and track_count > 0:
        info += f" | Tracks: {track_count} [{tracker_tag}]"
    if cfg.SHOW_FPS:
        info += f" | FPS: {fps:.1f}" if fps > 0 else " | FPS: --"
    if cfg.USE_ADAPTIVE_CONF:
        info += f" | Conf: {adaptive_conf:.2f}"
    info += f" | {device_tag}"

    (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(img, (10, 10), (20 + tw, 40), (0, 0, 0), -1)
    cv2.putText(img, info, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)