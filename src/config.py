# Output dimensions
# 1280x720, 1920x1080, 2560x1440 
OUTPUT_W = 2560
OUTPUT_H = 1440

# YOLO face detector config
YOLO_WEIGHTS = "models/yolov8n-face-lindevs.pt"  # YOLO face weights here
YOLO_CONF = 0.35             # confidence threshold
YOLO_IOU = 0.5               # NMS IoU threshold
YOLO_IMG_SIZE = 640          # inference size
YOLO_DEVICE = "mps"          # "cpu", "cuda", "mps"

# Portrait aspect ratios allowed (w:h), cycle via hotkey
ASPECT_CANDIDATES = [(1,1), (2,3), (3,4)]

# Grid policy
MAX_VISIBLE_PORTRAITS = 8
ONE_ROW_MAX = 4    # use 1 row up to this many people, else 2 rows
MIN_FACE_PX = 45    # minimum face height (px) to accept a detection

# Composition rules
HEADROOM_RATIO = 1.0/6.0     # ~16.7% headroom above crown
FOUR_H_MULT = 2.0            # ~head-to-chest ratio for cropping (multiplier of face height)

# Tracker
IOU_MATCH_THRESHOLD = 0.3    # IOU threshold
TRACK_FORGET_T = 30          # frames to keep a track alive without match
CENTER_DIST_PX = 160         # maximum center distance to consider a match

# Camera — single-cam mode
CAMERA_INDEX = 0              # 0-2 indexes, check with camera_test.py

# Dual-camera stitching (set DCS_ENABLED = True to enable)
DCS_ENABLED = True            # True = use two cameras stitched into a panorama
DCS_CAM0 = 0                  # left camera index
DCS_CAM1 = 1                  # right camera index
DCS_W = 1280                  # capture width per camera
DCS_H = 720                   # capture height per camera
DCS_FPS = 30
DCS_MAX_FEATURES = 1500       # ORB max features
DCS_RATIO = 0.75              # Lowe ratio
DCS_MIN_AGREE = 20            # minimum ORB agreement
DCS_WINDOWED = False          # use windowed ORB (faster)
DCS_WINDOW_FRAC = 0.5         # fraction of frame for windowed ORB
DCS_SMOOTH_ALPHA = 0.2        # EMA for dx
DCS_AUTO_SEAM = True          # automatically find optimal seam position in overlap zone
DCS_SEAM_FRAC = 0.5           # manual seam position as fraction of overlap (used when AUTO_SEAM=False)
DCS_BLEND_WIDTH = 10          # feather-blend width (px) on each side of the seam for smooth transition
DCS_COLOR_CORRECT = True      # enable histogram-based colour correction between left/right cameras
DCS_CC_STRENGTH = 0.8         # colour correction blend strength (0.0=none, 1.0=full correction)
DCS_RECALC_EVERY = 1          # re-estimate dx every N frames (0 = lock)
DCS_DEDUP_IOU = 0.15          # IoU threshold for overlap face dedup
DCS_DEDUP_MAX_DX = 80.0       # max horizontal centre distance for dedup

# ASD
ASD_ON_THR = 0.15             # lip/jaw-open score threshold to turn ON “speaking”.
ASD_OFF_THR = 0.10            # score threshold to turn OFF “speaking”.
ASD_EMA_ALPHA = 0.40          # ex moving avg, controls how much smoothing is applied to the raw lip/jaw-open signal.
ASD_SUSTAIN = 8               # How many frames to “hold” speaking = True after a drop.
SHOW_SPEAK_SCORE = False       # enable speak score hud

# Panoramic mini-window
PANO_ENABLED = True          # Enable/disable the panoramic preview window
PANO_WIDTH = 480             # Width of the expanded panoramic window (px)
PANO_HEIGHT = 270            # Height of the expanded panoramic window (px)
PANO_COLLAPSED_H = 18        # Height when collapsed, showing only the arrow strip (px)
PANO_MARGIN_BOTTOM = 24      # Distance from the bottom of the main frame (px)

PANO_EXPAND_MS = 220         # Animation duration for expansion (milliseconds)
PANO_COLLAPSE_MS = 160       # Animation duration for collapse (milliseconds)
PANO_HOVER_RADIUS = 72       # Hover detection radius around the arrow to trigger expansion (px)
PANO_AUTO_HIDE_SEC = 3.0     # Auto-collapse delay after hover ends (seconds)

PANO_BORDER_PX = 2           # Thickness of the border around the panoramic window (px)
PANO_RADIUS_PX = 10          # Corner radius for rounded edges (px)
PANO_BG_COLOR = (24, 24, 24) # Background fill color (BGR)
PANO_BORDER_COLOR = (64, 64, 64) # Border line color (BGR)
PANO_ARROW_COLOR = (180, 180, 180) # Color of the collapse/expand arrow (BGR)
PANO_SHADOW = True           # Whether to render a drop shadow behind the panoramic window