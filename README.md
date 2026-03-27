Loading ../weights/best_ncnn_model for NCNN inference...
/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/trackers/byte_tracker.py:202: RuntimeWarning: divide by zero encountered in scalar divide
  ret[2] /= ret[3]
/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/trackers/byte_tracker.py:202: RuntimeWarning: invalid value encountered in scalar divide
  ret[2] /= ret[3]
/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/trackers/byte_tracker.py:186: RuntimeWarning: invalid value encountered in scalar multiply
  ret[2] *= ret[3]
/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/trackers/utils/kalman_filter.py:153: RuntimeWarning: invalid value encountered in dot
  mean = np.dot(self._update_mat, mean)
2026-03-27 12:20:30,627 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/trackers/utils/kalman_filter.py:191: RuntimeWarning: invalid value encountered in dot
  mean = np.dot(mean, self._motion_mat.T)
2026-03-27 12:20:31,011 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
2026-03-27 12:20:31,445 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
2026-03-27 12:20:31,860 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
2026-03-27 12:20:32,237 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
2026-03-27 12:20:32,632 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
2026-03-27 12:20:33,022 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
2026-03-27 12:20:33,373 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
2026-03-27 12:20:33,736 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
2026-03-27 12:20:34,118 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
2026-03-27 12:20:34,491 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
2026-03-27 12:20:34,839 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
2026-03-27 12:20:35,264 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
2026-03-27 12:20:35,657 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
2026-03-27 12:20:36,048 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
2026-03-27 12:20:36,440 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
2026-03-27 12:20:36,802 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
2026-03-27 12:20:37,175 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
2026-03-27 12:20:37,575 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
2026-03-27 12:20:37,975 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
2026-03-27 12:20:38,424 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
2026-03-27 12:20:39,079 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
2026-03-27 12:20:39,493 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
2026-03-27 12:20:39,897 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
2026-03-27 12:20:40,141 - INFO - Frame 50/-230584300921369: Det=47, Tracks=69, FPS=5.2 | Classes: Drone:47
2026-03-27 12:20:40,262 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
2026-03-27 12:20:40,654 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
^CTraceback (most recent call last):
  File "/home/vietuav/Drone-Detection-PI/AI-6/YOLOv8/main.py", line 76, in <module>
    process_video(model, device, logger)
    ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vietuav/Drone-Detection-PI/AI-6/YOLOv8/video_processor.py", line 63, in process_video
    _run_single_video(cap, os.path.basename(video_path), model, device, logger)
    ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vietuav/Drone-Detection-PI/AI-6/YOLOv8/video_processor.py", line 130, in _run_single_video
    frame, stats = fp.process_frame(
                   ~~~~~~~~~~~~~~~~^
        model, device, frame, frame_count, total_frames, current_fps, logger
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/vietuav/Drone-Detection-PI/AI-6/YOLOv8/frame_processor.py", line 395, in process_frame
    draw_targeting_overlay(annotated, targeting_dets)
    ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vietuav/Drone-Detection-PI/AI-6/YOLOv8/targeting_overlay.py", line 210, in draw_targeting_overlay
    _draw_zone(frame, locked=False)
    ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
  File "/home/vietuav/Drone-Detection-PI/AI-6/YOLOv8/targeting_overlay.py", line 132, in _draw_zone
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, ZONE_THICKNESS)
    ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt
ioctl(VIDIOC_QBUF): Bad file descriptor
