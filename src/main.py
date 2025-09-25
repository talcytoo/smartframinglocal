import cv2 as cv
import numpy as np
import time
from rich import print as rprint

from .framing.face_detect import FaceDetector
from .framing.tracker import StableTracker
from .framing.layout import plan_layout_and_crops, compose_canvas
from .framing.animation import LayoutAnimator

from .features.face_landmarker import FaceLandmarkerASD
from .features.face_reid import FaceReID, ReIDConfig

from .ui.pano_window import PanoWindow, PanoConfig
from .ui.mouse import MouseState
from . import config

# Adjustable parameters
SMOOTH_ALPHA = getattr(config, "SMOOTH_ALPHA", 0.35)
SPAWN_SHRINK = getattr(config, "SPAWN_SHRINK", 0.82)
BORDER_THICK = 6
SPEAKER_BORDER_COLOR = (80, 220, 80)  # BGR
NON_SPEAKER_BORDER_COLOR = None
SHOW_SPEAK_SCORE = getattr(config, "SHOW_SPEAK_SCORE", True)

# ReID debug panel state and tuning
REID_STATE = {
    "show": False,     # disabled at startup, toggle with 'd'
    "focus_tid": None, # manually selected track id (cycled with 'q' / 'w')
}
REID_DEBUG_PANEL_W = 240
REID_DEBUG_PANEL_H = 240
REID_DEBUG_MARGIN = 12
REID_DEBUG_TOPK = 5
PANEL_INNER_PAD = 6

def ema_bbox(prev, cur, alpha):
    if prev is None:
        return cur
    px, py, pw, ph = prev
    cx, cy, cw, ch = cur
    nx = int(px * alpha + cx * (1 - alpha))
    ny = int(py * alpha + cy * (1 - alpha))
    nw = int(pw * alpha + cw * (1 - alpha))
    nh = int(ph * alpha + ch * (1 - alpha))
    return nx, ny, max(1, nw), max(1, nh)

def shrink_slot_rect(x, y, w, h, factor):
    nw = max(1, int(w * factor))
    nh = max(1, int(h * factor))
    nx = x + (w - nw) // 2
    ny = y + (h - nh) // 2
    return nx, ny, nw, nh

def build_pano_cfg() -> PanoConfig:
    return PanoConfig(
        enabled=getattr(config, 'PANO_ENABLED', True),
        width=getattr(config, 'PANO_WIDTH', 480),
        height=getattr(config, 'PANO_HEIGHT', 270),
        collapsed_h=getattr(config, 'PANO_COLLAPSED_H', 18),
        margin_bottom=getattr(config, 'PANO_MARGIN_BOTTOM', 24),
        expand_ms=getattr(config, 'PANO_EXPAND_MS', 220),
        collapse_ms=getattr(config, 'PANO_COLLAPSE_MS', 160),
        hover_radius=getattr(config, 'PANO_HOVER_RADIUS', 72),
        auto_hide_sec=float(getattr(config, 'PANO_AUTO_HIDE_SEC', 3.0)),
        border_px=getattr(config, 'PANO_BORDER_PX', 2),
        radius_px=getattr(config, 'PANO_RADIUS_PX', 10),
        bg_color=getattr(config, 'PANO_BG_COLOR', (24, 24, 24)),
        border_color=getattr(config, 'PANO_BORDER_COLOR', (64, 64, 64)),
        arrow_color=getattr(config, 'PANO_ARROW_COLOR', (180, 180, 180)),
        shadow=getattr(config, 'PANO_SHADOW', True),
    )

def reid_panel_rect(canvas_h):
    x0 = REID_DEBUG_MARGIN
    y0 = canvas_h - REID_DEBUG_PANEL_H - REID_DEBUG_MARGIN
    x1 = x0 + REID_DEBUG_PANEL_W
    y1 = y0 + REID_DEBUG_PANEL_H
    return x0, y0, x1, y1

def tracked_list_from_plan(plan):
    if not plan:
        return []
    ids = [s.get('track_id') for s in plan.get('slots', []) if s.get('track_id') is not None]
    seen, ordered = set(), []
    for tid in ids:
        if tid not in seen:
            seen.add(tid)
            ordered.append(tid)
    return ordered

def draw_reid_debug(canvas, reid, tracked_order, prefer_tid=None):
    if not REID_STATE["show"] or canvas is None:
        return canvas

    Hc, Wc = canvas.shape[:2]
    x0, y0, x1, y1 = reid_panel_rect(Hc)

    target_tid = None
    if REID_STATE["focus_tid"] in tracked_order:
        target_tid = REID_STATE["focus_tid"]
    elif prefer_tid in tracked_order:
        target_tid = prefer_tid
    elif tracked_order:
        target_tid = tracked_order[0]
    else:
        return canvas

    dbg = reid.get_debug(target_tid)
    if not dbg:
        return canvas

    cv.rectangle(canvas, (x0-2, y0-2), (x1+2, y1+2), (30, 30, 30), -1)
    cv.rectangle(canvas, (x0-2, y0-2), (x1+2, y1+2), (80, 80, 80), 1)

    inner_w = REID_DEBUG_PANEL_W - 2 * PANEL_INNER_PAD
    inner_h = REID_DEBUG_PANEL_H - 2 * PANEL_INNER_PAD
    pts = dbg.get('pts')
    if pts is not None and len(pts) > 0:
        for p in pts:
            px = int(x0 + PANEL_INNER_PAD + p[0] * max(inner_w - 1, 1))
            py = int(y0 + PANEL_INNER_PAD + p[1] * max(inner_h - 1, 1))
            cv.circle(canvas, (px, py), 1, (200, 200, 255), -1)

    info = reid.get_last_match_info(target_tid)
    if info:
        txt = f"match old {info['old']}  d={info['dist']:.2f}"
        cv.putText(canvas, txt, (x0 + PANEL_INNER_PAD, y1 - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.48, (140, 255, 140), 1, cv.LINE_AA)

    cv.putText(canvas, f"ReID debug  tid={target_tid}", (x0, y0 - 6),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv.LINE_AA)

    return canvas

def draw_static_pano_strip(canvas_bgr, full_frame_bgr, cfg: PanoConfig):
    Hc, Wc = canvas_bgr.shape[:2]
    ph = min(cfg.height, max(40, Hc // 3))
    pw = min(cfg.width, Wc - 40)
    x = (Wc - pw) // 2
    y = Hc - cfg.margin_bottom - ph

    cv.rectangle(canvas_bgr, (x, y), (x + pw, y + ph), cfg.bg_color, -1)
    cv.rectangle(canvas_bgr, (x, y), (x + pw, y + ph), cfg.border_color, cfg.border_px)

    fh, fw = full_frame_bgr.shape[:2]
    if fw > 0 and fh > 0:
        scale = min((pw - 8) / fw, (ph - 8) / fh)
        rw = max(1, int(fw * scale))
        rh = max(1, int(fh * scale))
        mini = cv.resize(full_frame_bgr, (rw, rh), interpolation=cv.INTER_AREA)
        x0 = x + (pw - rw) // 2
        y0 = y + (ph - rh) // 2
        canvas_bgr[y0:y0 + rh, x0:x0 + rw] = mini

    return canvas_bgr

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

    anim = LayoutAnimator(tau=0.20, shrink_in=0.9)

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

    reid = FaceReID(ReIDConfig(
        model_asset_path='models/face_landmarker.task',
        reid_window_sec=5.0,
        sim_threshold=0.18,
        use_blendshapes=True,
        lmk_norm=True,
    ))

    pano_cfg = build_pano_cfg()
    pano_mode = getattr(config, "PANO_MODE", "overlay").lower()  # "overlay" or "framing"
    if pano_mode not in ("overlay", "framing"):
        pano_mode = "overlay"

    rprint(f"[bold red]Smart Portrait Framing — v1.3[/]  [cyan]Pano mode: {pano_mode}[/]  toggle 'd' debug, cycle 'q'/'w', toggle pano mode 'm'")

    win_name = "Smart Portrait Framing (Demo)"
    cv.namedWindow(win_name, cv.WINDOW_NORMAL | cv.WINDOW_GUI_EXPANDED)
    mouse = MouseState()
    cv.setMouseCallback(win_name, mouse.callback)
    pano = None

    last_people = 0
    last_plan = None
    smooth_bboxes = {}
    prev_ids = set()
    fps_hist = []
    frame_idx = 0
    aspect_idx = 1
    use_rows_hint = None

    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            rprint("[red]Camera read failed.[/]")
            break

        detections = detector.detect(frame)

        ids = tracker.update([d["bbox"] for d in detections])

        annotated = []
        for d, tid in zip(detections, ids):
            if tid is not None:
                prev = smooth_bboxes.get(tid)
                sm = ema_bbox(prev, tuple(d["bbox"]), SMOOTH_ALPHA)
                smooth_bboxes[tid] = sm
                d["bbox"] = sm
                d["track_id"] = tid
            annotated.append(d)

        people_now = len([a for a in annotated if a.get("track_id") is not None])

        speaking_map = {}
        open_score_map = {}
        H, W = frame.shape[:2]
        pad_px = 12
        for a in annotated:
            tid = a.get("track_id")
            if tid is None:
                continue
            x, y, w, h = map(int, a["bbox"])
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

        cur_ids = {a['track_id'] for a in annotated if a.get('track_id') is not None}
        exited = prev_ids - cur_ids
        entered = cur_ids - prev_ids
        prev_ids = cur_ids.copy()

        for a in annotated:
            tid = a.get('track_id')
            if tid is None:
                continue
            x, y, w, h = map(int, a["bbox"])
            x0 = max(0, x - pad_px); y0 = max(0, y - pad_px)
            x1 = min(W, x + w + pad_px); y1 = min(H, y + h + pad_px)
            roi = frame[y0:y1, x0:x1]
            reid.update_track_embedding(tid, roi)

        for tid in exited:
            reid.note_exit(tid)

        id_remap = {}
        for tid in entered:
            old_id = reid.try_reassign(tid)
            if old_id is not None and old_id != tid:
                id_remap[tid] = old_id

        CAP = int(getattr(config, "MAX_VISIBLE_PORTRAITS",
                          getattr(config, "MAX_CELLS_PER_FRAME", 12)))

        def center_bias_score(bbox, W, H):
            x, y, w, h = bbox
            cx = x + 0.5 * w
            cy = y + 0.5 * h
            dx = cx - 0.5 * W
            dy = cy - 0.5 * H
            dist2 = dx * dx + dy * dy
            return 1.0 / (dist2 + 1.0)

        ranked = []
        for a in annotated:
            tid = a.get("track_id")
            if tid is None:
                continue
            x, y, w, h = map(int, a["bbox"])
            area = float(w * h)
            is_spk = 1.0 if speaking_map.get(tid, False) else 0.0
            cb = center_bias_score((x, y, w, h), W, H)
            score = (is_spk * 1e6) + (area * 1.0) + (cb * 1e3)
            ranked.append((tid, score))

        ranked.sort(key=lambda t: -t[1])
        keep_tids = {tid for tid, _ in ranked[:CAP]} if CAP > 0 else {tid for tid, _ in ranked}

        annotated = [a for a in annotated if a.get("track_id") in keep_tids]
        people_now = len(annotated)

        # If pano_mode == "framing", reserve a bottom strip and plan the layout above it
        reserve_h = 0
        if pano_mode == "framing" and pano_cfg.enabled:
            reserve_h = min(pano_cfg.height + pano_cfg.margin_bottom, config.OUTPUT_H // 3)

        if people_now > 0:
            rows_hint_local = use_rows_hint
            target_h = max(1, config.OUTPUT_H - reserve_h)
            plan = plan_layout_and_crops(
                frame,
                annotated,
                (config.OUTPUT_W, target_h),
                config.ASPECT_CANDIDATES[aspect_idx],
                config.ONE_ROW_MAX,
                config.HEADROOM_RATIO,
                config.FOUR_H_MULT,
                rows_hint_local
            )

            if last_plan and people_now > last_people:
                prev_ids_in_plan = {s.get("track_id") for s in last_plan["slots"] if s.get("track_id") is not None}
                cur_ids_in_plan = {s.get("track_id") for s in plan["slots"] if s.get("track_id") is not None}
                new_ids_in_plan = cur_ids_in_plan - prev_ids_in_plan
                if new_ids_in_plan:
                    for s in plan["slots"]:
                        tid = s.get("track_id")
                        if tid in new_ids_in_plan:
                            x, y, w, h = s["slot_xywh"]
                            s["slot_xywh"] = shrink_slot_rect(x, y, w, h, SPAWN_SHRINK)

            if id_remap:
                for s in plan.get('slots', []):
                    tid = s.get('track_id')
                    if tid in id_remap:
                        s['track_id'] = id_remap[tid]

            plan = anim.update(plan, dt=time.time() - t0)

            # Compose into top area (if reserving, we’ll paste into the top region later)
            canvas_top, _ = compose_canvas(frame, plan)

            if reserve_h > 0:
                canvas = np.zeros((config.OUTPUT_H, config.OUTPUT_W, 3), dtype=np.uint8)
                canvas[:config.OUTPUT_H - reserve_h, :, :] = canvas_top
            else:
                canvas = canvas_top

            for s in plan.get("slots", []):
                tid = s.get("track_id")
                if tid is None:
                    continue
                sx, sy, sw, sh = s["slot_xywh"]
                if speaking_map.get(tid, False):
                    cv.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), SPEAKER_BORDER_COLOR, BORDER_THICK)
                elif NON_SPEAKER_BORDER_COLOR is not None:
                    cv.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), NON_SPEAKER_BORDER_COLOR, 2)
                if SHOW_SPEAK_SCORE:
                    label = f"{open_score_map.get(tid, 0.0):.2f}"
                    cv.putText(canvas, label, (sx + 8, sy + 24),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)

            last_plan = plan
            last_people = people_now
        else:
            tracker.reset()
            smooth_bboxes.clear()
            last_plan = None
            last_people = 0
            canvas = cv.resize(frame, (config.OUTPUT_W, config.OUTPUT_H))

        # Pano behavior
        if pano_mode == "overlay":
            if speaking_map and pano is not None:
                pano.notify_activity(any(speaking_map.values()))
            if pano is None:
                pano = PanoWindow(canvas.shape[1], canvas.shape[0], pano_cfg)
            else:
                pano.set_canvas_size(canvas.shape[1], canvas.shape[0])
            pano.set_mouse(mouse.x, mouse.y)
            canvas = pano.update_and_draw(canvas, frame)
        else:
            # Pano as part of framing: draw a fixed strip in the reserved bottom area
            if pano_cfg.enabled:
                canvas = draw_static_pano_strip(canvas, frame, pano_cfg)

        # ReID debug panel (toggle 'd', cycle 'q' and 'w')
        tracked_order = tracked_list_from_plan(last_plan)
        prefer_tid = next((tid for tid, v in speaking_map.items() if v), None)
        canvas = draw_reid_debug(canvas, reid, tracked_order, prefer_tid)

        """
        # TEMP ID LABELS
        if last_plan is not None:
            remapped_values = set(id_remap.values()) if id_remap else set()
            for s in last_plan.get('slots', []):
                tid = s.get('track_id')
                if tid is None:
                    continue
                sx, sy, sw, sh = s['slot_xywh']
                label = f"ID {tid}"
                if tid in remapped_values:
                    label += "  ↺"
                cv.putText(canvas, label, (sx + 8, sy + 28),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)
        """

        dt = time.time() - t0
        fps = 1.0 / max(dt, 1e-3)
        fps_hist = (fps_hist + [fps])[-30:]
        hud_text = f"People: {people_now}  FPS: {np.mean(fps_hist):.1f}  Pano:{pano_mode}"
        if len(cur_ids) > people_now and people_now == min(len(cur_ids), CAP):
            hud_text += "  (capped)"
        hud = canvas.copy()
        cv.putText(hud, hud_text, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv.LINE_AA)

        cv.imshow(win_name, hud)
        frame_idx += 1
        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('e'):
            pano_cfg.enabled = not pano_cfg.enabled
            rprint(f"[yellow]Pano enabled: {pano_cfg.enabled}[/]")
        if key == ord('r'):
            pano_mode = "framing" if pano_mode == "overlay" else "overlay"
            pano = None
            rprint(f"[yellow]Pano mode switched to: {pano_mode}[/]")
        if key == ord('d'):
            REID_STATE["show"] = not REID_STATE["show"]
            if not REID_STATE["show"]:
                REID_STATE["focus_tid"] = None
        if key == ord('q'):
            if tracked_order:
                if REID_STATE["focus_tid"] not in tracked_order:
                    REID_STATE["focus_tid"] = tracked_order[0]
                else:
                    idx = tracked_order.index(REID_STATE["focus_tid"])
                    REID_STATE["focus_tid"] = tracked_order[(idx - 1) % len(tracked_order)]
                REID_STATE["show"] = True
        if key == ord('w'):
            if tracked_order:
                if REID_STATE["focus_tid"] not in tracked_order:
                    REID_STATE["focus_tid"] = tracked_order[0]
                else:
                    idx = tracked_order.index(REID_STATE["focus_tid"])
                    REID_STATE["focus_tid"] = tracked_order[(idx + 1) % len(tracked_order)]
                REID_STATE["show"] = True

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    setup()
