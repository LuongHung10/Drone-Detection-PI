"""
main.py - Entry point for YOLO Detection & Tracking System

Run:
    python main.py
"""
import os
import torch
import config as cfg

import hardware
hardware.apply_hardware_overrides()
device = hardware.select_device()

import logger as log_module
log_module.logger = log_module.setup_logger()
logger = log_module.logger


from ultralytics import YOLO

print(f"\n📦 Loading model: {cfg.MODEL_PATH}")
model = YOLO(cfg.MODEL_PATH)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

if hardware.IS_RASPBERRY_PI5 and not hardware.HAS_HAILO_NPU:
    try:
        model.model = torch.quantization.quantize_dynamic(
            model.model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print("✅ Model quantized to INT8 for Pi 5 CPU")
        if logger:
            logger.info("Model quantized to INT8")
    except Exception as e:
        print(f"⚠️  Quantization skipped: {e}")

print(f"\n📦 Model Info:")
print(f"   Classes  : {len(model.names)} → {list(model.names.values())}")
print(f"   Conf     : {cfg.CONF_THRESH}")
print(f"   Img size : {cfg.IMGSZ}")
print(f"   Tracker  : {cfg.TRACKER_TYPE if cfg.ENABLE_TRACKING else 'disabled'}")
print(f"   Device   : {device}")

if logger:
    logger.info(f"Model: {cfg.MODEL_PATH}")
    logger.info(f"  Classes: {list(model.names.values())}")
    logger.info(f"  Conf: {cfg.CONF_THRESH}, ImgSz: {cfg.IMGSZ}")
    logger.info(f"  Tracker: {cfg.TRACKER_TYPE if cfg.ENABLE_TRACKING else 'disabled'}")
    logger.info(f"  Device: {device}")

if __name__ == "__main__":
    from video_processor import process_video
    process_video(model, device, logger)