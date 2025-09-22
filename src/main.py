import cv2 as cv
import numpy as np
import time
from rich import print as rprint

from .framing.face_detect import FaceDetector
from .framing.tracker import StableTracker
from .framing.layout import plan_layout_and_crops, compose_canvas
from .framing.animation import LayoutAnimator
from .asd.face_landmarker import FaceLandmarkerASD
from . import config

# Adjustable params
SMOOTH_ALPHA = getattr(config, "SMOOTH_ALPHA", 0.35)      # bbox EMA (0..1)
SPAWN_SHRINK = getattr(config, "SPAWN_SHRINK", 0.82)      # initial size factor for new cells
BORDER_THICK = 6
SPEAKER_BORDER_COLOR = (80, 220, 80)                     # BGR green
NON_SPEAKER_BORDER_COLOR = None                          # None → no border

SHOW_SPEAK_SCORE = getattr(config, "SHOW_SPEAK_SCORE", True)

# Face Landmarker ASD
asd = FaceLandmarkerASD(
    model_asset_path="models/face_landmarker.task",
    running_mode="IMAGE",
    max_faces=1,
    use_blendshapes=True,
    ema_alpha=getattr(config, "ASD_EMA_ALPHA", 0.55),
    on_thr=getattr(config, "ASD_ON_THR", 0.35),
    off_thr=getattr(config, "ASD_OFF_THR", 0.22),
    sustain_frames=getattr(config, "ASD_SUSTAIN", 6),
    lips_fallback=True,
)


def _ema_bbox(prev, cur, alpha):
    # Exponential moving average for bbox (x,y,w,h)
    if prev is None:
        return cur
    px, py, pw, ph = prev
    cx, cy, cw, ch = cur
    one = 1.0 - alpha
    return (
        int(one * px + alpha * cx),
        int(one * py + alpha * cy),
        int(one * pw + alpha * cw),
        int(one * ph + alpha * ch),
    )


def _shrink_slot_rect(x, y, w, h, factor):
    # Shrink a rect about its center by 'factor' (0..1)
    nw = max(1, int(w * factor))
    nh = max(1, int(h * factor))
    nx = x + (w - nw) // 2
    ny = y + (h - nh) // 2
    return nx, ny, nw, nh


def setup():
    cap = cv.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {config.CAMERA_INDEX}")

    detector = FaceDetector(min_face_px=config.MIN_FACE_PX)
    tracker = StableTracker(
        match_iou_thr=config.IOU_MATCH_THRESHOLD,
        forget_after=config.TRACK_FORGET_T,
        enter_confirm=3,
        exit_confirm=10,
    )

    # Slower animator to reduce wobble
    anim = LayoutAnimator(tau=0.20, shrink_in=0.9)

    aspect_idx = 1        # default to 2:3
    use_rows_hint = None  # None → auto; else 1 or 2

    rprint("[bold red]Smart Portrait Framing — Single Camera Demo[/]")

    # State
    last_people = 0
    last_plan = None
    smooth_bboxes = {}   # tid -> smoothed (x,y,w,h)

    fps_hist = []
    frame_idx = 0

    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            rprint("[red]Camera read failed.[/]")
            break

        # Detect faces
        detections = detector.detect(frame)

        # Track ids (used for smoothing and spawn detection only)
        ids = tracker.update([d["bbox"] for d in detections])

        # Apply EMA per track id
        annotated = []
        for d, tid in zip(detections, ids):
            if tid is not None:
                prev = smooth_bboxes.get(tid)
                sm = _ema_bbox(prev, tuple(d["bbox"]), SMOOTH_ALPHA)
                smooth_bboxes[tid] = sm
                d["bbox"] = sm
                d["track_id"] = tid
            annotated.append(d)

        people_now = len([a for a in annotated if a.get("track_id") is not None])

        # Active speaker detection (per cropped ROI)
        speaking_map = {}    # track_id -> bool
        open_score_map = {}  # track_id -> float
        if people_now > 0:
            H, W = frame.shape[:2]
            pad_px = 12

            for a in annotated:
                tid = a.get("track_id")
                if tid is None:
                    continue
                x, y, w, h = map(int, a["bbox"])

                # Expand ROI slightly and clamp
                x0 = max(0, x - pad_px)
                y0 = max(0, y - pad_px)
                x1 = min(W, x + w + pad_px)
                y1 = min(H, y + h + pad_px)
                if x1 <= x0 or y1 <= y0:
                    speaking_map[tid] = False
                    open_score_map[tid] = 0.0
                    continue

                roi = frame[y0:y1, x0:x1]
                score, is_spk = asd.score_and_update(roi, track_id=tid)
                speaking_map[tid] = bool(is_spk)
                open_score_map[tid] = float(score)

        if people_now > 0:
            # For 4+ people, prefer 2 rows
            rows_hint_local = 2 if people_now >= 4 else use_rows_hint

            # Smart framing
            aspect_wh = config.ASPECT_CANDIDATES[aspect_idx]
            plan = plan_layout_and_crops(
                frame,
                annotated,
                out_size=(config.OUTPUT_W, config.OUTPUT_H),
                aspect_wh=aspect_wh,
                one_row_max=config.ONE_ROW_MAX,
                headroom_ratio=config.HEADROOM_RATIO,
                four_h_mult=config.FOUR_H_MULT,
                rows_hint=rows_hint_local,
            )

            # Spawn animation for new ids
            if last_plan and people_now > last_people:
                prev_ids = {s.get("track_id") for s in last_plan["slots"] if s.get("track_id") is not None}
                cur_ids = {s.get("track_id") for s in plan["slots"] if s.get("track_id") is not None}
                new_ids = cur_ids - prev_ids
                if new_ids:
                    new_slots = []
                    for s in plan["slots"]:
                        tid = s.get("track_id")
                        if tid in new_ids:
                            x, y, w, h = s["slot_xywh"]
                            s["slot_xywh"] = _shrink_slot_rect(x, y, w, h, SPAWN_SHRINK)
                        new_slots.append(s)
                    plan["slots"] = new_slots

            plan = anim.update(plan, dt=time.time() - t0)
            canvas, _ = compose_canvas(frame, plan)

            # Overlay (speaker border + score)
            for s in plan.get("slots", []):
                tid = s.get("track_id")
                if tid is None:
                    continue
                sx, sy, sw, sh = s["slot_xywh"]

                if speaking_map.get(tid, False):
                    cv.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), SPEAKER_BORDER_COLOR, BORDER_THICK)

                if SHOW_SPEAK_SCORE:
                    label = f"{open_score_map.get(tid, 0.0):.2f}"
                    cv.putText(
                        canvas,
                        label,
                        (sx + 8, sy + 24),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv.LINE_AA,
                    )

            last_plan = plan
            last_people = people_now
        else:
            # No people → reset and show raw camera
            tracker.reset()
            smooth_bboxes.clear()
            last_plan = None
            last_people = 0
            canvas = cv.resize(frame, (config.OUTPUT_W, config.OUTPUT_H))

        # HUD
        dt = time.time() - t0
        fps = 1.0 / max(dt, 1e-3)
        fps_hist = (fps_hist + [fps])[-30:]
        hud_text = f"People: {people_now}  FPS: {np.mean(fps_hist):.1f}"

        hud = canvas.copy()
        cv.putText(hud, hud_text, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv.LINE_AA)
        cv.imshow("Smart Portrait Framing (Demo)", hud)

        frame_idx += 1
        if cv.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    setup()
