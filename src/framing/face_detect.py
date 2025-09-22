import numpy as np
from typing import List, Dict, Tuple

from ultralytics import YOLO
from .. import config


class FaceDetector:
    # YOLO-based face detector (Ultralytics).
    # Expects a face-specific YOLO model (e.g., yolov8n-face.pt).
    # Returns dicts with "bbox", "score", and optional "keypoints".
    def __init__(self, min_face_px: int = 45):
        self.min_face_px = min_face_px

        # Load YOLO model
        self.model = YOLO(config.YOLO_WEIGHTS)

        # Runtime params
        self.conf = getattr(config, "YOLO_CONF", 0.35)
        self.iou = getattr(config, "YOLO_IOU", 0.5)
        self.imgsz = getattr(config, "YOLO_IMG_SIZE", 640)
        self.device = getattr(config, "YOLO_DEVICE", "cpu")

    @staticmethod
    def _xyxy_to_xywh(box: np.ndarray) -> Tuple[int, int, int, int]:
        # Convert xyxy box to xywh
        x1, y1, x2, y2 = box
        x, y = int(x1), int(y1)
        w, h = int(x2 - x1), int(y2 - y1)
        return x, y, max(w, 0), max(h, 0)

    def detect(self, frame_bgr: np.ndarray) -> List[Dict]:
        H, W = frame_bgr.shape[:2]

        # Run YOLO inference
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        out: List[Dict] = []
        if not results:
            return out

        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return out

        # Extract keypoints if available
        kp_obj = getattr(r, "keypoints", None)
        has_kp = kp_obj is not None and getattr(kp_obj, "xy", None) is not None
        kps = kp_obj.xy.cpu().numpy() if has_kp else None  # (N, K, 2)

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for i, b in enumerate(xyxy):
            x, y, w, h = self._xyxy_to_xywh(b)
            if h < self.min_face_px or w < self.min_face_px:
                continue

            score = float(confs[i]) if i < len(confs) else 0.0
            keypoints = {}

            if has_kp and i < len(kps):
                # Map common 5-keypoint format
                pts = kps[i]
                keypoints = {
                    "right_eye": (float(pts[0, 0]), float(pts[0, 1])),
                    "left_eye":  (float(pts[1, 0]), float(pts[1, 1])),
                    "nose":      (float(pts[2, 0]), float(pts[2, 1])),
                    "mouth_r":   (float(pts[3, 0]), float(pts[3, 1])),
                    "mouth_l":   (float(pts[4, 0]), float(pts[4, 1])),
                }

            out.append(
                {
                    "bbox": (x, y, w, h),
                    "score": score,
                    "keypoints": keypoints,
                }
            )

        return out