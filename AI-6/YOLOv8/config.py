"""
config.py - All configuration settings for YOLO Detection & Tracking System
"""
import logging

# =====================================================
# PATHS
# =====================================================
MODEL_PATH = "../weights/best3.pt"
VIDEO_PATH = "../video"
OUTPUT_DIR = "../results"

# =====================================================
# INFERENCE SETTINGS
# =====================================================
CONF_THRESH = 0.15
IOU_THRESH = 0.60
IMGSZ = 832

# Adaptive Image Size
USE_ADAPTIVE_IMGSZ = True
ADAPTIVE_IMGSZ_MIN = 720
ADAPTIVE_IMGSZ_MAX = 920
ADAPTIVE_IMGSZ_BASE = 832

# =====================================================
# MODEL & DEVICE
# =====================================================
MODEL_SIZE = "nano"     # "auto", "nano", "small", "medium", "large", "custom"
USE_HALF = True         # FP16 (requires GPU or NPU)
USE_TENSORRT = False    # Requires model export first
BATCH_SIZE = 16

# =====================================================
# TRACKING
# =====================================================
ENABLE_TRACKING = True
TRACKER_TYPE = "bytetrack"   # "bytetrack", "botsort", "kcf", "hybrid"
TRACK_BUFFER = 50
TRACK_CONF_THRESH = 0.30
TRACK_IOU_THRESH = 0.50
MIN_TRACK_FRAMES = 2
MAX_AGE = 50

# =====================================================
# TEMPORAL SMOOTHING
# =====================================================
USE_TEMPORAL_SMOOTHING = True
SMOOTHING_ALPHA = 0.8
SMOOTHING_HISTORY = 5

# =====================================================
# KALMAN FILTER
# =====================================================
USE_KALMAN_FILTER = True
KALMAN_PROCESS_NOISE = 0.03
KALMAN_MEASUREMENT_NOISE = 0.3

# =====================================================
# ADAPTIVE CONFIDENCE
# =====================================================
USE_ADAPTIVE_CONF = True
ADAPTIVE_CONF_MIN = 0.05
ADAPTIVE_CONF_MAX = 0.20
ADAPTIVE_CONF_STEP = 0.01

# =====================================================
# HUNGARIAN MATCHING
# =====================================================
USE_HUNGARIAN_MATCHING = True
HUNGARIAN_IOU_THRESH = 0.5

# =====================================================
# OCCLUSION HANDLING
# =====================================================
USE_OCCLUSION_HANDLING = True
OCCLUSION_THRESH = 0.3
OCCLUSION_BUFFER = 10

# =====================================================
# MULTI-SCALE DETECTION
# =====================================================
USE_MULTI_SCALE = False
USE_SMART_MULTI_SCALE = True
MULTI_SCALE_FACTORS = [0.8, 1.0, 1.2]
MULTI_SCALE_THRESHOLD = 2
MULTI_SCALE_FRAME_INTERVAL = 3

# =====================================================
# NCNN INFERENCE
# =====================================================
USE_NCNN = False             # Set True to use NCNN instead of PyTorch
# NCNN model path — export first:
#   yolo export model=best.pt format=ncnn opset=12
#   This creates a folder: best_ncnn_model/
NCNN_MODEL_PATH = "../weights/best_ncnn_model"

# =====================================================
# FRAME SKIPPING
# =====================================================
USE_FRAME_SKIP = True    
FRAME_SKIP_N   = 3
# SKIP_THRESHOLD_STABLE = 0.98
# MAX_SKIP_FRAMES = 0
# SKIP_ON_NO_DETECTIONS = False
# SKIP_CONSECUTIVE_NO_DET = 10

# =====================================================
# ADVANCED OPTIMIZATIONS
# =====================================================
USE_PREPROCESSING_CACHE = True
USE_POSTPROCESSING_OPTIMIZATION = True
OPTIMIZE_GPU_MEMORY = True
USE_EARLY_EXIT = True
USE_BATCH_PROCESSING = False
ASYNC_INFERENCE = False

# =====================================================
# MEMORY MANAGEMENT
# =====================================================
ENABLE_MEMORY_OPTIMIZATION = False      # Overridden per hardware in setup.py
MEMORY_CLEANUP_INTERVAL = 60

# =====================================================
# DISPLAY
# =====================================================
SHOW_FPS = True
DEBUG_DETECTIONS = False
SAVE_RESULT = True
WINDOW_NAME = "🚀 YOLO Segmentation Viewer"

# =====================================================
# PERFORMANCE MONITORING
# =====================================================
PERFORMANCE_MONITORING = True

# =====================================================
# LOGGING
# =====================================================
ENABLE_LOGGING = True
LOG_TO_FILE = True
LOG_LEVEL = logging.INFO
LOG_DETECTION_STATS = True
LOG_TRACKING_STATS = True
LOG_EVERY_N_FRAMES = 50