# Smart Portrait Framing — Single Camera Demo Ver.

This demo takes a webcam feed and automatically turns it into a clean and organized grid of cropped human portraits. It detects faces, tracks individuals over time, frames each person into a consistent aspect ratio, and highlights the active speakers in real time. The project aims to facilitate complex in-person group discussions where multiple people share a single computer, such as in study groups or classrooms.

---

## Features
This project combines several components to deliver stable and natural framing.  

It uses a YOLO face model for detection, paired with a lightweight IoU-based tracker to maintain consistent IDs across frames.  
Faces are cropped into uniform portrait rectangles with headroom adjustments, then arranged automatically into one or two rows depending on group size.  

To reduce jitter, exponential moving average smoothing is applied to both bounding boxes and layout transitions.  
MediaPipe’s Face Landmarker provides visual-only speaker detection, allowing the system to highlight the active speaker without relying on audio input.  

---

## How It Works
1. Frames are read from the webcam using OpenCV.  
2. YOLO locates bounding boxes around visible faces.  
3. A lightweight tracker preserves consistent IDs across frames.  
4. Bounding boxes are smoothed to avoid sudden jumps.  
5. MediaPipe estimates lip activity to flag when someone is speaking.  
6. A layout planner assigns people to portrait slots based on count and aspect ratio.  
7. The system composites the portraits, draws highlights for speakers, and overlays relevant HUD info. 

---

## Getting Started

### 1. Clone repo and set up the environment
```bash
git clone https://github.com/yourname/smart-portrait-framing.git
cd smart-portrait-framing
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. Run the demo
```bash
python -m src.main
```
