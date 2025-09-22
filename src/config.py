# Output dimensions
# 1280x720, 1920x1080
OUTPUT_W = 1920
OUTPUT_H = 1080

# YOLO face detector config
YOLO_WEIGHTS = "models/yolov8n-face-lindevs.pt"  # put your YOLO face weights here
YOLO_CONF = 0.35                          # confidence threshold
YOLO_IOU = 0.5                            # NMS IoU threshold
YOLO_IMG_SIZE = 640                       # inference size
YOLO_DEVICE = "mps"                       # "cpu" or "cuda" (or "mps" on Apple Silicon)

# Portrait aspect ratios allowed (w:h) — will cycle via hotkey
ASPECT_CANDIDATES = [(1,1), (2,3), (3,4)]

# Grid policy
MAX_CELLS_PER_FRAME = 12
ONE_ROW_MAX = 4     # use 1 row up to this many people, else 2 rows
MIN_FACE_PX = 45    # minimum face height (px) to accept a detection

# Composition rules
HEADROOM_RATIO = 1.0/6.0         # ~16.7% headroom above crown
FOUR_H_MULT = 4.0            # ~head-to-waist = ~4× face height

# Tracker
IOU_MATCH_THRESHOLD = 0.3    # IOU threshold
TRACK_FORGET_T = 30          # frames to keep a track alive without match
CENTER_DIST_PX = 160         # maximum center distance to consider a match

# Camera
CAMERA_INDEX = 0              # 0-2 indexes, check with camera_test.py

# ASD
ASD_ON_THR = 0.15             # lip/jaw-open score threshold to turn ON “speaking”.
ASD_OFF_THR = 0.10            # score threshold to turn OFF “speaking”.
ASD_EMA_ALPHA = 0.40          # ex moving avg, controls how much smoothing is applied to the raw lip/jaw-open signal.
ASD_SUSTAIN = 8               # How many frames to “hold” speaking = True after a drop.
SHOW_SPEAK_SCORE = True       # enable speak score hud