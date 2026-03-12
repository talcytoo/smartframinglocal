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

from .camera.dualcamstitch import DualCamStitcher, DualCamConfig
from .camera.face_dedup import deduplicate_overlap_faces

SMOOTH_ALPHA = getattr(config, "SMOOTH_ALPHA", 0.35)
SPAWN_SHRINK = getattr(config, "SPAWN_SHRINK", 0.82)
BORDER_THICK = 6
SPEAKER_BORDER_COLOR = (80, 220, 80)
NON_SPEAKER_BORDER_COLOR = None
SHOW_SPEAK_SCORE = getattr(config, "SHOW_SPEAK_SCORE", True)

REID_STATE = {
    "show": False,
    "focus_tid": None,
}
REID_DEBUG_PANEL_W = 240
REID_DEBUG_PANEL_H = 240
REID_DEBUG_MARGIN = 12
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
def compute_pano_dims(frame_w: int, frame_h: int) -> tuple:
    max_pano_w = int(frame_w * 0.3)
    max_pano_h = int(frame_h * 0.3)
    return max_pano_w, max_pano_h
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
    side_margin = max(24, Wc // 20)
    max_pw = Wc - 2 * side_margin
    ph_from_max = int(max_pw / 2.5)
    ph_cap = max(40, int(Hc * 0.42))
    ph = min(ph_from_max, ph_cap)
    pw = int(ph * 2.5)

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

    return canvas_bgr, (ph + cfg.margin_bottom)

def compute_viewport_y(plan, output_h: int, visible_h: int) -> int:
    if not plan or "slots" not in plan or not plan["slots"]:
        return 0
    centers = []
    for s in plan["slots"]:
        if "slot_xywh" in s:
            sx, sy, sw, sh = s["slot_xywh"]
            centers.append(sy + sh * 0.5)
    if not centers:
        return 0
    centers = sorted(centers)
    mid = centers[len(centers)//2] if len(centers) % 2 == 1 else 0.5*(centers[len(centers)//2 - 1] + centers[len(centers)//2])
    half = visible_h * 0.5
    y0 = int(round(mid - half))
    y0 = max(0, min(y0, output_h - visible_h))
    return y0

def _build_dual_cam_cfg() -> DualCamConfig:
    return DualCamConfig(
        cam0_index=getattr(config, 'DCS_CAM0', 0),
        cam1_index=getattr(config, 'DCS_CAM1', 1),
        width=getattr(config, 'DCS_W', 1280),
        height=getattr(config, 'DCS_H', 720),
        fps=getattr(config, 'DCS_FPS', 30),
        max_features=getattr(config, 'DCS_MAX_FEATURES', 1500),
        ratio=getattr(config, 'DCS_RATIO', 0.75),
        min_agree=getattr(config, 'DCS_MIN_AGREE', 20),
        windowed=getattr(config, 'DCS_WINDOWED', False),
        window_frac=getattr(config, 'DCS_WINDOW_FRAC', 0.5),
        smooth_alpha=getattr(config, 'DCS_SMOOTH_ALPHA', 0.2),
        auto_seam=getattr(config, 'DCS_AUTO_SEAM', True),
        seam_frac=getattr(config, 'DCS_SEAM_FRAC', 0.5),
        blend_width=getattr(config, 'DCS_BLEND_WIDTH', 10),
        color_correct=getattr(config, 'DCS_COLOR_CORRECT', True),
        cc_strength=getattr(config, 'DCS_CC_STRENGTH', 0.8),
        recalc_every=getattr(config, 'DCS_RECALC_EVERY', 1),
    )

def setup():
    use_dual = getattr(config, 'DCS_ENABLED', False)

    stitcher = None
    cap = None
    if use_dual:
        dc_cfg = _build_dual_cam_cfg()
        stitcher = DualCamStitcher(dc_cfg)
        stitcher.start()
        rprint("[bold cyan]Dual-cam stitch mode ON[/]")
    else:
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
    pano_mode = "embed"

    mode_tag = " [dual-cam]" if use_dual else ""
    rprint(f"[bold red]Smart Portrait Framing — v1.4{mode_tag}[/]  toggle 'd' debug, cycle 'q'/'w', toggle pano enable 'e', toggle mode 'r' (overlay/embed)")

    win_name = "Smart Portrait Framing (Demo)"
    cv.namedWindow(win_name, cv.WINDOW_NORMAL | cv.WINDOW_GUI_EXPANDED)
    cv.resizeWindow(win_name, config.OUTPUT_W, config.OUTPUT_H)
    cv.moveWindow(win_name, 0, 0)

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
    pano_dims_adjusted = False

    while True:
        t0 = time.time()

        if use_dual:
            ok, frame = stitcher.read()
        else:
            ok, frame = cap.read()
        if not ok:
            if not use_dual:
                rprint("[red]Camera read failed.[/]")
                break
            time.sleep(0.005)
            continue

        if not pano_dims_adjusted and frame is not None:
            H, W = frame.shape[:2]
            pano_w, pano_h = compute_pano_dims(W, H)
            pano_cfg.width = pano_w
            pano_cfg.height = pano_h
            pano_dims_adjusted = True

        detections = detector.detect(frame)

        if use_dual and stitcher.has_dx:
            dedup_iou = getattr(config, 'DCS_DEDUP_IOU', 0.15)
            dedup_dx = getattr(config, 'DCS_DEDUP_MAX_DX', 80.0)
            os_, oe = stitcher.overlap_range
            detections = deduplicate_overlap_faces(
                detections,
                stitch_dx=stitcher.dx,
                cam_width=stitcher.cam_width,
                overlap_start=os_,
                overlap_end=oe,
                iou_thr=dedup_iou,
                max_center_dx=dedup_dx,
            )

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

        people_now_raw = len([a for a in annotated if a.get("track_id") is not None])

        speaking_map = {}
        open_score_map = {}
        H, W = frame.shape[:2]
        pad_px = 12
        for a in annotated:
            tid = a.get("track_id")
            if tid is None:
                continue
            x, y, w, h = map(int, a["bbox"])
            x0 = max(0, x - pad_px); y0 = max(0, y - pad_px)
            x1 = min(W, x + w + pad_px); y1 = min(H, y + h + pad_px)
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

        CAP_default = int(getattr(config, "MAX_VISIBLE_PORTRAITS",
                                  getattr(config, "MAX_CELLS_PER_FRAME", 12)))
        CAP = CAP_default
        if pano_mode == "embed":
            CAP = min(CAP_default, 4)

        def center_bias_score(bbox, W, H):
            x, y, w, h = bbox
            cx = x + 0.5 * w; cy = y + 0.5 * h
            dx = cx - 0.5 * W; dy = cy - 0.5 * H
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

        reserve_h = 0
        if pano_mode == "embed" and pano_cfg.enabled:
            ideal_ph = int(config.OUTPUT_W / 2.5)
            ph_cap = int(config.OUTPUT_H * 0.45)
            pano_h = min(max(40, ideal_ph), ph_cap)
            reserve_h = pano_h + pano_cfg.margin_bottom

        if people_now > 0:
            rows_hint_local = 1 if pano_mode == "embed" else use_rows_hint
            if pano_mode == "embed":
                aspect_for_plan = getattr(config, "FRAMING_PORTRAIT_ASPECT", (9, 16))
            else:
                aspect_for_plan = config.ASPECT_CANDIDATES[aspect_idx]

            layout_h = config.OUTPUT_H - reserve_h if pano_mode == "embed" and reserve_h > 0 else config.OUTPUT_H
            plan = plan_layout_and_crops(
                frame,
                annotated,
                (config.OUTPUT_W, layout_h),
                aspect_for_plan,
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

            full_canvas, _ = compose_canvas(frame, plan)

            if pano_mode == "embed" and reserve_h > 0:
                top_h = max(1, config.OUTPUT_H - reserve_h)

                canvas = np.zeros((config.OUTPUT_H, config.OUTPUT_W, 3), dtype=np.uint8)
                canvas[:top_h, :, :] = full_canvas[:top_h, :, :]

                for s in plan.get("slots", []):
                    tid = s.get("track_id")
                    if tid is None:
                        continue
                    sx, sy, sw, sh = s["slot_xywh"]
                    vx0 = max(0, sx)
                    vy0 = max(0, sy)
                    vx1 = min(config.OUTPUT_W - 1, sx + sw - 1)
                    vy1 = min(top_h - 1, sy + sh - 1)
                    if vx1 > vx0 and vy1 > vy0:
                        if speaking_map.get(tid, False):
                            cv.rectangle(canvas, (vx0, vy0), (vx1, vy1), SPEAKER_BORDER_COLOR, BORDER_THICK)
                        elif NON_SPEAKER_BORDER_COLOR is not None:
                            cv.rectangle(canvas, (vx0, vy0), (vx1, vy1), NON_SPEAKER_BORDER_COLOR, 2)
                        if SHOW_SPEAK_SCORE:
                            tx = vx0 + 8
                            ty = max(12, vy0 + 24)
                            cv.putText(canvas, f"{open_score_map.get(tid, 0.0):.2f}",
                                       (tx, ty), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                                       (255, 255, 255), 2, cv.LINE_AA)
            else:
                canvas = full_canvas
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
                        cv.putText(canvas, f"{open_score_map.get(tid, 0.0):.2f}",
                                   (sx + 8, sy + 24), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                                   (255, 255, 255), 2, cv.LINE_AA)

            last_plan = plan
            last_people = people_now
        else:
            tracker.reset()
            smooth_bboxes.clear()
            last_plan = None
            last_people = 0
            canvas = cv.resize(frame, (config.OUTPUT_W, config.OUTPUT_H))

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
            if pano_cfg.enabled:
                canvas, _ = draw_static_pano_strip(canvas, frame, pano_cfg)

        tracked_order = tracked_list_from_plan(last_plan)
        prefer_tid = next((tid for tid, v in speaking_map.items() if v), None)
        canvas = draw_reid_debug(canvas, reid, tracked_order, prefer_tid)

        dt = time.time() - t0
        fps = 1.0 / max(dt, 1e-3)
        fps_hist = (fps_hist + [fps])[-30:]
        cam_tag = "DualCam" if use_dual else "SingleCam"
        hud_text = f"People: {people_now}  FPS: {np.mean(fps_hist):.1f}  {cam_tag}  Mode:{pano_mode}"
        if use_dual and stitcher.dx is not None:
            hud_text += f"  dx:{stitcher.dx}"
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
            pano_mode = "embed" if pano_mode == "overlay" else "overlay"
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

    if use_dual and stitcher:
        stitcher.stop()
    if cap:
        cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    setup()