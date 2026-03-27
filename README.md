Lock released
💾 Saving output to: ../results/output_video0
qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in "/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/cv2/qt/plugins"
QFontDatabase: Cannot find font directory /home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/cv2/qt/fonts.
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.

🎮 Controls:  SPACE = pause/resume   ESC / Q = quit

Loading ../weights/best_ncnn_model for NCNN inference...
/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/trackers/byte_tracker.py:202: RuntimeWarning: divide by zero encountered in scalar divide
  ret[2] /= ret[3]
/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/trackers/byte_tracker.py:202: RuntimeWarning: invalid value encountered in scalar divide
  ret[2] /= ret[3]
/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/trackers/byte_tracker.py:186: RuntimeWarning: invalid value encountered in scalar multiply
  ret[2] *= ret[3]
/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/trackers/utils/kalman_filter.py:153: RuntimeWarning: invalid value encountered in dot
  mean = np.dot(self._update_mat, mean)
2026-03-27 11:28:51,077 - WARNING - Tracker error (resetting): 1-th leading minor of the array is not positive definite
Traceback (most recent call last):
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
  File "/home/vietuav/Drone-Detection-PI/AI-6/YOLOv8/frame_processor.py", line 254, in process_frame
    x1, y1, x2, y2 = map(int, box)
    ^^^^^^^^^^^^^^
ValueError: cannot convert float NaN to integer
ioctl(VIDIOC_QBUF): Bad file descriptor
