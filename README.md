# Smart Portrait Framing (DTEN Design Proposal)

This project involves taking one or several camera feeds and automatically turning them into a clean and organized grid of cropped human portraits. It detects faces, tracks individuals over time, frames each person into a consistent aspect ratio, and highlight the active speakers in real time. This demonstration version is a design proposal demo for DTEN, aiming to facilitate complex in-person group discussions where multiple people share a single device, such as in study groups, classrooms, or conference rooms. 

---

## Features
This project combines several components to deliver stable and natural framing.  

It uses a YOLO face model for detection, paired with a lightweight IoU-based tracker to maintain consistent IDs across frames. Faces are cropped into uniform portrait rectangles with headroom adjustments, then arranged automatically into one or two rows depending on group size.  

To reduce jitter, exponential moving average smoothing is applied to both bounding boxes and layout transitions. MediaPipe’s Face Landmarker provides visual-only speaker detection, allowing the system to highlight the active speakers without relying on audio input.  

Additional features include a collapsible panoramic mini-window that displays the entire camera view and short-term face re-identification, which enables the recognition of individuals who briefly leave and return with the same ID.  

---

## How It Works
1. Frames are read from a connected webcam using OpenCV.  
2. YOLO locates bounding boxes around visible faces.  
3. A lightweight tracker preserves consistent IDs across frames.  
4. Bounding boxes are smoothed to avoid sudden jumps.  
5. MediaPipe estimates lip activity to flag when someone is speaking.  
6. A layout planner assigns people to portrait slots based on count and aspect ratio.  
7. The system composites the portraits, draws highlights for speakers, and overlays relevant HUD info.  
8. The panoramic mini-window shows the full camera view, expanding on hover and collapsing when idle.  
9. Short-term re-identification links new detections with recently exited tracks if they return quickly.  

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

### (Optional) Debug toggle
During runtime, press d to toggle the ReID debug panel, and use q/w to cycle through tracked IDs.
