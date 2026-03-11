# Smart Portrait Framing

**Smart Portrait Framing (DTEN Design Proposal)**

This project involves taking one or several camera feeds and automatically turning them into a clean and organized grid of cropped human portraits. It detects faces, tracks individuals over time, frames each person into a consistent aspect ratio, and highlights active speakers in real-time. This demonstration version is a design proposal demo for DTEN, aiming to facilitate complex in-person group discussions where multiple people share a single device, such as in study groups, classrooms, or conference rooms.

---

## Project Structure

```
smartframinglocal/
├── models/
│   ├── face_landmarker.task          # MediaPipe face landmarks (ASD + ReID)
│   └── yolov8n-face-lindevs.pt      # YOLO face detection weights
├── src/
│   ├── main.py                       # Entry point & main loop
│   ├── config.py                     # All tuneable parameters
│   ├── camera/
│   │   ├── dualcamstitch.py          # Dual-camera ORB stitcher
│   │   └── face_dedup.py             # Overlap-zone face deduplication
│   ├── framing/
│   │   ├── face_detect.py            # YOLO face detector wrapper
│   │   ├── tracker.py                # IoU-based face tracker
│   │   ├── layout.py                 # Portrait crop & grid layout planner
│   │   └── animation.py              # Smooth layout transitions
│   ├── features/
│   │   ├── face_landmarker.py        # MediaPipe active speaker detection
│   │   └── face_reid.py              # Short-term face re-identification
│   └── ui/
│       ├── pano_window.py            # Collapsible panoramic mini-window
│       └── mouse.py                  # Mouse state handler
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- macOS (Apple Silicon MPS), Linux, or Windows with CUDA
- Two USB cameras (dual-cam mode) or one webcam (single-cam mode)

### Python Dependencies

```bash
pip install -r requirements.txt
```

Key packages: `opencv-python`, `ultralytics`, `mediapipe`, `facenet-pytorch`, `numpy`, `rich`

## Getting Started

### 1. Clone and set up the environment

```bash
cd smartframinglocal
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the demo

```bash
python -m src.main
```

### 3. Configuration

All parameters are in `src/config.py`. Key settings:

| Setting | Description | Default |
|---------|-------------|---------|
| `OUTPUT_W` / `OUTPUT_H` | Output window resolution | 2560x1440 |
| `YOLO_DEVICE` | Inference device (`cpu`, `cuda`, `mps`) | `mps` |
| `DCS_ENABLED` | Enable dual-camera stitching | `True` |
| `DCS_CAM0` / `DCS_CAM1` | Left / right camera index | 0 / 1 |
| `DCS_W` / `DCS_H` | Capture resolution per camera | 1280x720 |
| `FOUR_H_MULT` | Portrait crop height as multiplier of face height | 2.0 |
| `MAX_VISIBLE_PORTRAITS` | Max simultaneous portrait slots | 4 |
| `ASPECT_CANDIDATES` | Cycleable portrait aspect ratios (w:h) | (1,1), (2,3), (3,4) |

#### Dual-Camera Stitch Settings (DCS)

| Setting | Description | Default |
|---------|-------------|---------|
| `DCS_MAX_FEATURES` | Max ORB features for alignment | 1500 |
| `DCS_RATIO` | Lowe ratio test threshold | 0.75 |
| `DCS_MIN_AGREE` | Minimum matching keypoints | 20 |
| `DCS_SMOOTH_ALPHA` | EMA smoothing for horizontal shift | 0.2 |
| `DCS_AUTO_SEAM` | Auto-find optimal seam in overlap zone | `True` |
| `DCS_SEAM_FRAC` | Manual seam position (when AUTO_SEAM=False) | 0.5 |
| `DCS_BLEND_WIDTH` | Feather-blend width (px) at the seam | 10 |
| `DCS_COLOR_CORRECT` | Histogram-based colour correction | `True` |
| `DCS_CC_STRENGTH` | Colour correction strength (0.0-1.0) | 0.8 |
| `DCS_RECALC_EVERY` | Re-estimate shift every N frames (0=lock) | 1 |
| `DCS_DEDUP_IOU` | IoU threshold for overlap face dedup | 0.15 |

## Runtime Controls

| Key | Action |
|-----|--------|
| `ESC` | Quit |
| `e` | Toggle panoramic mini-window on/off |
| `r` | Switch between overlay and embed pano modes |
| `d` | Toggle ReID debug panel |
| `q` / `w` | Cycle through tracked IDs in debug panel |

## Implementation Notes

- Captures from one or two cameras; dual-cam mode stitches them into a panorama via ORB feature matching
- YOLO detects face bounding boxes; an IoU tracker assigns stable IDs across frames
- In dual-cam mode, duplicate detections in the camera overlap zone are removed
- Bounding boxes are EMA-smoothed to reduce jitter
- MediaPipe face landmarks estimate lip activity for visual-only speaker detection
- Portraits are ranked by speaking status, face area, and centre bias; the top N are kept
- A layout planner assigns each person to a grid slot with uniform crop sizing and headroom
- Layout transitions are animated with exponential decay for smooth motion
- The panoramic mini-window shows the full camera view — hover to expand, or use embed mode
- Short-term face re-identification links returning individuals to their previous track ID

## References

- OpenCV: https://opencv.org/
- Ultralytics YOLO: https://docs.ultralytics.com/
- MediaPipe Face Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
- FaceNet PyTorch: https://github.com/timesler/facenet-pytorch
- ORB Feature Detection: https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html
