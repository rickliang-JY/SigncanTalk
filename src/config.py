import os

MODEL_ID = os.environ.get("SIGNCAN_MODEL_ID", "google/gemma-4-E4B-it")

VIDEO_FPS = 1
MAX_FRAMES = 60
FRAME_RESIZE_LONG_EDGE = 896

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.3
TOP_P = 0.9

DTYPE = os.environ.get("SIGNCAN_DTYPE", "bfloat16")
DEVICE_MAP = os.environ.get("SIGNCAN_DEVICE_MAP", "auto")
