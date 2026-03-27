Loading ../weights/best_ncnn_model for NCNN inference...
/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/trackers/byte_tracker.py:202: RuntimeWarning: divide by zero encountered in scalar divide
  ret[2] /= ret[3]
/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/trackers/byte_tracker.py:202: RuntimeWarning: invalid value encountered in scalar divide
  ret[2] /= ret[3]
/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/trackers/byte_tracker.py:186: RuntimeWarning: invalid value encountered in scalar multiply
  ret[2] *= ret[3]
/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/trackers/utils/kalman_filter.py:153: RuntimeWarning: invalid value encountered in dot
  mean = np.dot(self._update_mat, mean)
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
  File "/home/vietuav/Drone-Detection-PI/AI-6/YOLOv8/frame_processor.py", line 132, in process_frame
    results = model.track(
        frame,
    ...<11 lines>...
        stream=False,
    )
  File "/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/engine/model.py", line 579, in track
    return self.predict(source=source, stream=stream, **kwargs)
           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/engine/model.py", line 536, in predict
    return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)
                                                                    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/engine/predictor.py", line 225, in __call__
    return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Results into one
  File "/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/torch/utils/_contextlib.py", line 40, in generator_context
    response = gen.send(None)
  File "/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/engine/predictor.py", line 342, in stream_inference
    self.run_callbacks("on_predict_postprocess_end")
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/engine/predictor.py", line 513, in run_callbacks
    callback(self)
    ~~~~~~~~^^^^^^
  File "/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/trackers/track.py", line 93, in on_predict_postprocess_end
    tracks = tracker.update(det, result.orig_img, getattr(result, "feats", None))
  File "/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/trackers/byte_tracker.py", line 365, in update
    unconfirmed[itracked].update(detections[idet], self.frame_id)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/trackers/byte_tracker.py", line 165, in update
    self.mean, self.covariance = self.kalman_filter.update(
                                 ~~~~~~~~~~~~~~~~~~~~~~~~~^
        self.mean, self.covariance, self.convert_coords(new_tlwh)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/trackers/utils/kalman_filter.py", line 219, in update
    chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
                         ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/scipy/_lib/_util.py", line 1181, in wrapper
    return f(*arrays, *other_args, **kwargs)
  File "/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/scipy/linalg/_decomp_cholesky.py", line 183, in cho_factor
    c, lower = _cholesky(a, lower=lower, overwrite_a=overwrite_a, clean=False,
               ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         check_finite=check_finite)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/scipy/linalg/_decomp_cholesky.py", line 39, in _cholesky
    raise LinAlgError(
        f"{info}-th leading minor of the array is not positive definite"
    )
numpy.linalg.LinAlgError: 1-th leading minor of the array is not positive definite
ioctl(VIDIOC_QBUF): Bad file descriptor
