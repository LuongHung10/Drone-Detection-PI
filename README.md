🍓 Raspberry Pi 5 detected - Optimizing for Pi 5 hardware...
⚠️  AI HAT NPU not detected - Using CPU inference
⚙️  Config optimized for Pi 5 CPU (no NPU)
💾 System RAM: 7.87 GB
🍓 Raspberry Pi 5 CPU (4 threads)
2026-03-27 16:08:44,575 - INFO - ============================================================
2026-03-27 16:08:44,575 - INFO - YOLO Detection & Tracking System Started
2026-03-27 16:08:44,575 - INFO - ============================================================

📦 Loading model: ../weights/best_ncnn_model
WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'.

📦 Loading NCNN model: ../weights/best_ncnn_model
WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'.
✅ NCNN model loaded — running on CPU (ARM optimised)

📦 Model Info:
Loading ../weights/best_ncnn_model for NCNN inference...
Loading ../weights/best_ncnn_model for NCNN inference...
   Classes  : 1 → ['Drone']
   Conf     : 0.15
   Img size : 576
   Tracker  : bytetrack
   Device   : cpu
2026-03-27 16:08:48,307 - INFO - Model: ../weights/best_ncnn_model
Loading ../weights/best_ncnn_model for NCNN inference...
2026-03-27 16:08:48,370 - INFO -   Classes: ['Drone']
2026-03-27 16:08:48,370 - INFO -   Conf: 0.15, ImgSz: 576
2026-03-27 16:08:48,370 - INFO -   Tracker: bytetrack
2026-03-27 16:08:48,370 - INFO -   Device: cpu
📹 video0: 720x576 @ 25fps (live)
2026-03-27 16:08:50,723 - INFO - ============================================================
2026-03-27 16:08:50,724 - INFO - Processing: video0
2026-03-27 16:08:50,724 - INFO -   Resolution: 720x576, FPS: 25, Frames: -230584300921369
2026-03-27 16:08:50,724 - INFO - ============================================================
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
SHAPE: torch.Size([0, 6])
DATA: tensor([], size=(0, 6))
CONF: tensor([])
CLS : tensor([])
SHAPE: torch.Size([5, 6])
DATA: tensor([[549.8438, 464.5312, 720.0000, 576.0000,   1.0000,   0.0000],
        [612.6562, 460.6250, 720.0000, 576.0000,   1.0000,   0.0000],
        [704.6875, 471.0938, 720.0000, 576.0000,   1.0000,   0.0000],
        [662.5000, 453.2812, 720.0000, 576.0000,   1.0000,   0.0000],
        [  0.0000, 462.6562, 168.7500, 576.0000,   1.0000,   0.0000]])
CONF: tensor([1., 1., 1., 1., 1.])
CLS : tensor([0., 0., 0., 0., 0.])
SHAPE: torch.Size([5, 6])
DATA: tensor([[549.8438, 464.5312, 720.0000, 576.0000,   1.0000,   0.0000],
        [612.6562, 460.6250, 720.0000, 576.0000,   1.0000,   0.0000],
        [704.6875, 471.0938, 720.0000, 576.0000,   1.0000,   0.0000],
        [662.5000, 453.2812, 720.0000, 576.0000,   1.0000,   0.0000],
        [  0.0000, 462.6562, 168.7500, 576.0000,   1.0000,   0.0000]])
CONF: tensor([1., 1., 1., 1., 1.])
CLS : tensor([0., 0., 0., 0., 0.])
SHAPE: torch.Size([5, 6])
DATA: tensor([[549.8438, 464.5312, 720.0000, 576.0000,   1.0000,   0.0000],
        [612.6562, 460.6250, 720.0000, 576.0000,   1.0000,   0.0000],
        [704.6875, 471.0938, 720.0000, 576.0000,   1.0000,   0.0000],
        [662.5000, 453.2812, 720.0000, 576.0000,   1.0000,   0.0000],
        [  0.0000, 462.6562, 168.7500, 576.0000,   1.0000,   0.0000]])
CONF: tensor([1., 1., 1., 1., 1.])
CLS : tensor([0., 0., 0., 0., 0.])
SHAPE: torch.Size([5, 6])
DATA: tensor([[549.8438, 464.5312, 720.0000, 576.0000,   1.0000,   0.0000],
        [612.6562, 460.6250, 720.0000, 576.0000,   1.0000,   0.0000],
        [704.6875, 471.0938, 720.0000, 576.0000,   1.0000,   0.0000],
        [662.5000, 453.2812, 720.0000, 576.0000,   1.0000,   0.0000],
        [  0.0000, 462.6562, 168.7500, 576.0000,   1.0000,   0.0000]])
CONF: tensor([1., 1., 1., 1., 1.])
CLS : tensor([0., 0., 0., 0., 0.])
SHAPE: torch.Size([5, 6])
DATA: tensor([[  0.0000, 348.0000, 200.8125, 563.2500,   1.0000,   0.0000],
        [ 53.4375, 346.3125, 266.2500, 562.6875,   1.0000,   0.0000],
        [102.1406, 429.5625, 292.3594, 564.9375,   1.0000,   0.0000],
        [126.4688, 350.8125, 337.7812, 562.6875,   1.0000,   0.0000],
        [185.7188, 349.3125, 394.7812, 564.1875,   1.0000,   0.0000]])
CONF: tensor([1., 1., 1., 1., 1.])
CLS : tensor([0., 0., 0., 0., 0.])
SHAPE: torch.Size([5, 6])
DATA: tensor([[  0.0000, 348.0000, 200.8125, 563.2500,   1.0000,   0.0000],
        [ 53.4375, 346.3125, 266.2500, 562.6875,   1.0000,   0.0000],
        [102.1406, 429.5625, 292.3594, 564.9375,   1.0000,   0.0000],
        [126.4688, 350.8125, 337.7812, 562.6875,   1.0000,   0.0000],
        [185.7188, 349.3125, 394.7812, 564.1875,   1.0000,   0.0000]])
CONF: tensor([1., 1., 1., 1., 1.])
CLS : tensor([0., 0., 0., 0., 0.])
SHAPE: torch.Size([5, 6])
DATA: tensor([[  0.0000, 348.0000, 200.8125, 563.2500,   1.0000,   0.0000],
        [ 53.4375, 346.3125, 266.2500, 562.6875,   1.0000,   0.0000],
        [102.1406, 429.5625, 292.3594, 564.9375,   1.0000,   0.0000],
        [126.4688, 350.8125, 337.7812, 562.6875,   1.0000,   0.0000],
        [185.7188, 349.3125, 394.7812, 564.1875,   1.0000,   0.0000]])
CONF: tensor([1., 1., 1., 1., 1.])
CLS : tensor([0., 0., 0., 0., 0.])
SHAPE: torch.Size([5, 6])
DATA: tensor([[  0.0000, 348.0000, 200.8125, 563.2500,   1.0000,   0.0000],
        [ 53.4375, 346.3125, 266.2500, 562.6875,   1.0000,   0.0000],
        [102.1406, 429.5625, 292.3594, 564.9375,   1.0000,   0.0000],
        [126.4688, 350.8125, 337.7812, 562.6875,   1.0000,   0.0000],
        [185.7188, 349.3125, 394.7812, 564.1875,   1.0000,   0.0000]])
CONF: tensor([1., 1., 1., 1., 1.])
CLS : tensor([0., 0., 0., 0., 0.])
SHAPE: torch.Size([5, 6])
DATA: tensor([[  0.0000, 348.0000, 200.8125, 563.2500,   1.0000,   0.0000],
        [ 53.4375, 346.3125, 266.2500, 562.6875,   1.0000,   0.0000],
        [102.1406, 429.5625, 292.3594, 564.9375,   1.0000,   0.0000],
        [126.4688, 350.8125, 337.7812, 562.6875,   1.0000,   0.0000],
        [185.7188, 349.3125, 394.7812, 564.1875,   1.0000,   0.0000]])
CONF: tensor([1., 1., 1., 1., 1.])
CLS : tensor([0., 0., 0., 0., 0.])
SHAPE: torch.Size([5, 6])
DATA: tensor([[  0.0000, 348.0000, 200.8125, 563.2500,   1.0000,   0.0000],
        [ 53.4375, 346.3125, 266.2500, 562.6875,   1.0000,   0.0000],
        [102.1406, 429.5625, 292.3594, 564.9375,   1.0000,   0.0000],
        [126.4688, 350.8125, 337.7812, 562.6875,   1.0000,   0.0000],
        [185.7188, 349.3125, 394.7812, 564.1875,   1.0000,   0.0000]])
CONF: tensor([1., 1., 1., 1., 1.])
CLS : tensor([0., 0., 0., 0., 0.])
SHAPE: torch.Size([5, 6])
DATA: tensor([[  0.0000, 348.0000, 200.8125, 563.2500,   1.0000,   0.0000],
        [ 53.4375, 346.3125, 266.2500, 562.6875,   1.0000,   0.0000],
        [102.1406, 429.5625, 292.3594, 564.9375,   1.0000,   0.0000],
        [126.4688, 350.8125, 337.7812, 562.6875,   1.0000,   0.0000],
        [185.7188, 349.3125, 394.7812, 564.1875,   1.0000,   0.0000]])
CONF: tensor([1., 1., 1., 1., 1.])
CLS : tensor([0., 0., 0., 0., 0.])
SHAPE: torch.Size([5, 6])
DATA: tensor([[  0.0000, 348.0000, 201.0000, 563.2500,   1.0000,   0.0000],
        [ 53.4375, 346.3125, 266.2500, 562.6875,   1.0000,   0.0000],
        [102.3281, 429.7500, 292.1719, 564.7500,   1.0000,   0.0000],
        [126.4688, 350.8125, 337.7812, 562.6875,   1.0000,   0.0000],
        [186.0000, 349.3125, 394.8750, 564.1875,   1.0000,   0.0000]])
CONF: tensor([1., 1., 1., 1., 1.])
CLS : tensor([0., 0., 0., 0., 0.])
SHAPE: torch.Size([5, 6])
DATA: tensor([[  0.0000, 336.0000, 198.7500, 558.0000,   1.0000,   0.0000],
        [ 39.1875, 336.1875, 234.5625, 576.0000,   1.0000,   0.0000],
        [ 65.8125, 333.0000, 276.0000, 570.0000,   1.0000,   0.0000],
        [102.2812, 436.3125, 303.4688, 576.0000,   1.0000,   0.0000],
        [126.3750, 360.0000, 338.6250, 576.0000,   1.0000,   0.0000]])
CONF: tensor([1., 1., 1., 1., 1.])
CLS : tensor([0., 0., 0., 0., 0.])
SHAPE: torch.Size([5, 6])
DATA: tensor([[  0.0000, 342.7500, 200.2500, 576.0000,   1.0000,   0.0000],
        [ 30.9375, 348.7500, 241.1250, 558.0000,   1.0000,   0.0000],
        [ 90.3750, 372.7500, 300.3750, 570.0000,   1.0000,   0.0000],
        [102.1875, 434.8125, 301.3125, 576.0000,   1.0000,   0.0000],
        [126.1875, 364.6875, 337.6875, 576.0000,   1.0000,   0.0000]])
CONF: tensor([1., 1., 1., 1., 1.])
CLS : tensor([0., 0., 0., 0., 0.])
SHAPE: torch.Size([5, 6])
DATA: tensor([[  0.0000, 342.1875, 200.2500, 557.8125,   1.0000,   0.0000],
        [ 18.5625, 403.6875, 220.5000, 576.0000,   1.0000,   0.0000],
        [ 42.1875, 333.7500, 234.7500, 576.0000,   1.0000,   0.0000],
        [ 78.3750, 359.2500, 283.1250, 570.0000,   1.0000,   0.0000],
        [102.5625, 435.1875, 303.9375, 576.0000,   1.0000,   0.0000]])
CONF: tensor([1., 1., 1., 1., 1.])
CLS : tensor([0., 0., 0., 0., 0.])
^CTraceback (most recent call last):
  File "/home/vietuav/Drone-Detection-PI/AI-6/YOLOv8/main.py", line 82, in <module>
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
  File "/home/vietuav/Drone-Detection-PI/AI-6/YOLOv8/frame_processor.py", line 154, in process_frame
    results = model.predict(
        frame,
    ...<9 lines>...
        stream=False,
    )
  File "/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/engine/model.py", line 536, in predict
    return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)
                                                                    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/engine/predictor.py", line 225, in __call__
    return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Results into one
  File "/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/torch/utils/_contextlib.py", line 40, in generator_context
    response = gen.send(None)
  File "/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/engine/predictor.py", line 334, in stream_inference
    preds = self.inference(im, *args, **kwargs)
  File "/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/engine/predictor.py", line 182, in inference
    return self.model(im, augment=self.args.augment, visualize=visualize, embed=self.args.embed, *args, **kwargs)
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/nn/autobackend.py", line 273, in forward
    y = self.backend.forward(im, **forward_kwargs)
  File "/home/vietuav/Drone-Detection-PI/venv/lib/python3.13/site-packages/ultralytics/nn/backends/ncnn.py", line 71, in forward
    y = [np.array(ex.extract(x)[1])[None] for x in sorted(self.net.output_names())]
                  ~~~~~~~~~~^^^
KeyboardInterrupt
ioctl(VIDIOC_QBUF): Bad file descriptor
