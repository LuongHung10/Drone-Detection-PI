"""
video_processor.py - Video and camera processing pipeline
"""
import os
import time
from collections import defaultdict

import cv2

import config as cfg
import perf_monitor
import frame_processor as fp
import targeting_overlay as tgt


# =====================================================
# PUBLIC ENTRY POINT
# =====================================================
def process_video(model, device: str, logger=None):
    """
    Main entry point: processes VIDEO_PATH (file, folder, or camera index).

    Args:
        model:  Loaded YOLO model instance
        device: Device string ("cuda" / "cpu")
        logger: Optional logging.Logger
    """
    video_path = cfg.VIDEO_PATH

    # ── Camera input ─────────────────────────────────────────────────────────
    if video_path in ("0", 0):
        print("📹 Using camera input...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera")
            return
        _run_single_video(cap, "camera", model, device, logger)
        return

    # ── Folder of videos ──────────────────────────────────────────────────────
    if os.path.isdir(video_path):
        print(f"📁 Processing folder: {video_path}")
        exts = (".mp4", ".avi", ".mov", ".mkv")
        files = [f for f in os.listdir(video_path) if f.lower().endswith(exts)]
        if not files:
            print("❌ No video files found in folder")
            return
        for filename in files:
            full_path = os.path.join(video_path, filename)
            print(f"\n🚀 Processing: {filename}")
            cap = cv2.VideoCapture(full_path)
            if not cap.isOpened():
                print(f"❌ Cannot open: {filename}")
                continue
            _run_single_video(cap, filename, model, device, logger)
        return

    # ── Single video file ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Cannot open video: {video_path}")
        return
    _run_single_video(cap, os.path.basename(video_path), model, device, logger)


# =====================================================
# INTERNAL: SINGLE VIDEO / STREAM LOOP
# =====================================================
def _run_single_video(cap, video_name: str, model, device: str, logger=None):
    """Process one open VideoCapture until exhaustion or user exit."""
    fps_src = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if total_frames > 0:
        print(f"📹 {video_name}: {width}x{height} @ {fps_src}fps, {total_frames} frames")
    else:
        print(f"📹 {video_name}: {width}x{height} @ {fps_src}fps (live)")

    if logger:
        logger.info("=" * 60)
        logger.info(f"Processing: {video_name}")
        logger.info(f"  Resolution: {width}x{height}, FPS: {fps_src}, Frames: {total_frames}")
        logger.info("=" * 60)

    # ── Reset per-video state ─────────────────────────────────────────────────
    fp.reset_state()
    perf_monitor.reset()

    # ── Output video writer ───────────────────────────────────────────────────
    writer = None
    if cfg.SAVE_RESULT:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(cfg.OUTPUT_DIR, f"output_{video_name}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps_src, (width, height))
        print(f"💾 Saving output to: {out_path}")

    # ── Window ────────────────────────────────────────────────────────────────
    cv2.namedWindow(cfg.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cfg.WINDOW_NAME, 1280, 720)

    print("\n🎮 Controls:  SPACE = pause/resume   ESC / Q = quit\n")

    # ── Per-session stats ─────────────────────────────────────────────────────
    frame_count = 0
    current_fps = 0.0
    frame_times: list[float] = []
    video_start = time.time()
    paused = False
    frame = None

    total_detections = 0
    frames_with_det = 0
    frames_no_det = 0
    class_stats: dict[str, int] = defaultdict(int)
    all_track_ids: set[int] = set()

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Video processing complete.")
                break

            frame_count += 1
            t0 = time.time()
            frame, stats = fp.process_frame(
                model, device, frame, frame_count, total_frames, current_fps, logger
            )
            elapsed = time.time() - t0

            # Running FPS estimate
            frame_times.append(elapsed)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_t = sum(frame_times) / len(frame_times)
            current_fps = 1.0 / avg_t if avg_t > 0 else 0.0

            # Accumulate session stats
            if stats['detection_count'] > 0:
                frames_with_det += 1
                total_detections += stats['detection_count']
                for cls, cnt in stats['class_counts'].items():
                    class_stats[cls] += cnt
            else:
                frames_no_det += 1

            all_track_ids.update(stats['track_ids'])

            if writer:
                writer.write(frame)

        # Display
        if frame is not None:
            cv2.imshow(cfg.WINDOW_NAME, frame)

        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key in (27, ord('q')):
            break
        if key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")
        if key in (ord('x'), ord('X')):
            tgt.confirm_lock()
        if key == 13:   # Enter — release lock
            tgt.release_lock()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # ── Final summary ─────────────────────────────────────────────────────────
    total_time = time.time() - video_start
    avg_fps = frame_count / total_time if total_time > 0 else 0.0
    perf = perf_monitor.get_summary()

    print("\n" + "=" * 60)
    print("📊 SESSION STATISTICS")
    print("=" * 60)
    print(f"   Frames processed : {frame_count}")
    print(f"   Processing time  : {total_time:.2f}s")
    print(f"   Average FPS      : {avg_fps:.2f}")

    if perf:
        print("\n⚡ PERFORMANCE METRICS:")
        print(f"   Avg inference : {perf['avg_inference_ms']:.2f} ms")
        print(f"   Avg total     : {perf['avg_total_ms']:.2f} ms")
        print(f"   Min inference : {perf['min_inference_ms']:.2f} ms")
        print(f"   Max inference : {perf['max_inference_ms']:.2f} ms")
        print(f"   Est. FPS      : {perf['fps']:.2f}")

    print("=" * 60)

    if logger:
        logger.info("=" * 60)
        logger.info("FINAL STATISTICS")
        logger.info(f"  Frames: {frame_count}, Time: {total_time:.2f}s, FPS: {avg_fps:.2f}")
        if cfg.LOG_DETECTION_STATS:
            logger.info(f"  Frames w/ detections: {frames_with_det}")
            logger.info(f"  Frames w/o detections: {frames_no_det}")
            for cls, cnt in sorted(class_stats.items(), key=lambda x: -x[1]):
                logger.info(f"    {cls}: {cnt}")
        if cfg.LOG_TRACKING_STATS and cfg.ENABLE_TRACKING:
            logger.info(f"  Unique track IDs: {len(all_track_ids)}")
        logger.info("=" * 60)