from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2 as cv


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def pick_rows_cols(n_people: int, out_w: int, out_h: int, one_row_max: int, rows_hint: Optional[int]) -> Tuple[int, int]:
    if rows_hint in (1, 2):
        rows = rows_hint
    else:
        rows = 1 if n_people <= one_row_max else 2
    cols = int(np.ceil(n_people / rows)) if n_people > 0 else 1
    return rows, cols


def cell_size(out_w: int, out_h: int, rows: int, cols: int) -> Tuple[int, int]:
    cell_w = out_w // cols
    cell_h = out_h // rows
    return cell_w, cell_h


# Portrait bounds helper for uniform look
def _bounds_11_to_32(cell_w: int, cell_h: int) -> Tuple[int, int, int]:
    # Compute portrait bounds inside a slot (cell_w × cell_h) for aspect range [1:1 .. 3:2].
    minH = min(cell_h, cell_w)                       # 1:1 fit
    ar_32 = 3.0 / 2.0                                # 3:2 aspect ratio
    maxH = min(cell_h, int(cell_w / ar_32))          # 3:2 fit by height
    maxW = min(cell_w, int(ar_32 * cell_h))          # 3:2 fit by width
    return int(minH), int(maxH), int(maxW)


def compute_target_face_height(dets: List[Dict], cell_h: int, headroom_ratio: float) -> int:
    if not dets:
        return 0
    min_face = int(min(d["bbox"][3] for d in dets))  # smallest bbox height
    max_face = int((1 - headroom_ratio) * cell_h * (1 / 4)) * 4
    return min(min_face, int((5 / 6) * (cell_h / 4)) * 4)


def ensure_single_face_crop(src_shape, face_bbox, other_bboxes, face_h, aspect_wh, headroom_ratio, four_h_mult):
    # Multi-person helper for single-face crop
    H, W = src_shape[:2]
    x, y, w, h = face_bbox
    cx = x + w / 2.0
    top_head = y
    target_h = int(clamp(four_h_mult * face_h, face_h, H))
    headroom = int(headroom_ratio * target_h)
    crop_y1 = int(top_head - headroom)
    crop_y2 = crop_y1 + target_h

    aspect = aspect_wh[0] / aspect_wh[1]
    target_w = int(target_h * aspect)

    crop_x1 = int(cx - target_w / 2)
    crop_x2 = crop_x1 + target_w

    # Clamp to image bounds
    dx1 = max(0 - crop_x1, 0)
    dy1 = max(0 - crop_y1, 0)
    dx2 = max(crop_x2 - W, 0)
    dy2 = max(crop_y2 - H, 0)
    crop_x1 += dx1 - dx2
    crop_x2 += dx1 - dx2
    crop_y1 += dy1 - dy2
    crop_y2 += dy1 - dy2

    crop = [int(crop_x1), int(crop_y1), int(crop_x2 - crop_x1), int(crop_y2 - crop_y1)]

    def center(bb):
        return bb[0] + bb[2] / 2.0, bb[1] + bb[3] / 2.0

    # Avoid intruding faces
    for ob in other_bboxes:
        ocx, ocy = center(ob)
        inside = (crop_x1 <= ocx <= crop_x2) and (crop_y1 <= ocy <= crop_y2)
        if inside:
            shift = int(0.15 * target_w)
            if ocx < cx:
                crop_x1 = max(0, crop_x1 - shift)
                crop_x2 = crop_x1 + target_w
            else:
                crop_x2 = min(W, crop_x2 + shift)
                crop_x1 = crop_x2 - target_w
            crop = [int(crop_x1), int(crop_y1), int(crop_x2 - crop_x1), int(crop_y2 - crop_y1)]
    return crop


def _single_person_plan(frame_bgr, det, out_w, out_h, aspect_wh, headroom_ratio):
    H_src, W_src = frame_bgr.shape[:2]
    x, y, w, h = det["bbox"]
    aspect = aspect_wh[0] / aspect_wh[1]

    # Option A: fit by height (pillarbox left/right)
    a_h = out_h
    a_w = int(a_h * aspect)
    a_x = (out_w - a_w) // 2
    a_y = 0
    area_a = a_w * a_h if a_w <= out_w else -1  # invalidate if overflow

    # Option B: fit by width (letterbox top/bottom)
    b_w = out_w
    b_h = int(b_w / aspect)
    b_x = 0
    b_y = (out_h - b_h) // 2
    area_b = b_w * b_h if b_h <= out_h else -1  # invalidate if overflow

    # Choose the one with less black bars (larger area)
    if area_b > area_a:
        dst_w, dst_h, dst_x, dst_y = b_w, b_h, b_x, b_y
    else:
        dst_w, dst_h, dst_x, dst_y = a_w, a_h, a_x, a_y

    # Body-inclusive crop height (same as before)
    H_spec = (6.0 / 5.0) * h
    crop_h = int(min(4.0 * H_spec, float(H_src)))
    crop_w = int(crop_h * aspect)

    # Vertical placement with ~1/6 headroom
    headroom = (1.0 / 6.0) * crop_h
    top_head = y
    crop_y1 = int(top_head - headroom)
    crop_y1 = clamp(crop_y1, 0, max(0, H_src - crop_h))

    # Horizontal placement centered on face
    cx = x + 0.5 * w
    crop_x1 = int(cx - 0.5 * crop_w)
    crop_x1 = clamp(crop_x1, 0, max(0, W_src - crop_w))

    slot = {
        "slot_xywh": (int(dst_x), int(dst_y), int(dst_w), int(dst_h)),
        "crop_xywh": (int(crop_x1), int(crop_y1), int(crop_w), int(crop_h)),
        "track_id": det.get("track_id", None),
        "bbox": det["bbox"],
        "score": det.get("score", 0.0),
    }
    return slot


def plan_layout_and_crops(frame_bgr, detections: List[Dict], out_size: Tuple[int, int], aspect_wh: Tuple[int, int],
                          one_row_max: int, headroom_ratio: float, four_h_mult: float, rows_hint: Optional[int]):
    out_w, out_h = out_size
    n = len(detections)

    # Single-person: full portrait with pillarbox
    if n == 1:
        slot = _single_person_plan(frame_bgr, detections[0], out_w, out_h, aspect_wh, headroom_ratio)
        return {
            "rows": 1, "cols": 1,
            "cell_w": out_w, "cell_h": out_h,
            "slots": [slot],
            "aspect_wh": aspect_wh,
            "target_face_h": detections[0]["bbox"][3],
            "out_w": out_w, "out_h": out_h,
        }

    # Multi-person uniform layout
    rows, cols = pick_rows_cols(n, out_w, out_h, one_row_max, rows_hint)
    cell_w, cell_h = cell_size(out_w, out_h, rows, cols)

    target_aspect = aspect_wh[0] / aspect_wh[1]
    dst_h_fit = min(cell_h, int(cell_w / target_aspect))
    dst_w_fit = int(dst_h_fit * target_aspect)

    minFaceHeight = min(d["bbox"][3] for d in detections) if detections else 0
    targetFaceHeight = int(min(minFaceHeight, (5.0 / 6.0) * dst_h_fit))

    def downscale_for(face_h: int) -> float:
        return targetFaceHeight / max(face_h, 1e-6)

    ordered = sorted(detections, key=lambda d: (d.get("track_id", 1e9), d["bbox"][0]))
    slots = []
    H_src, W_src = frame_bgr.shape[:2]

    for idx, d in enumerate(ordered):
        row = idx // cols
        col = idx % cols

        slot_x = col * cell_w
        slot_y = row * cell_h
        dst_w, dst_h = dst_w_fit, dst_h_fit
        dst_x = slot_x + (cell_w - dst_w) // 2
        dst_y = slot_y + (cell_h - dst_h) // 2

        x, y, w, h = d["bbox"]
        scale = downscale_for(h)

        H0_raw = int(dst_h / max(scale, 1e-6))
        H_min = int((6.0 / 5.0) * h)
        H_max = int(min((6.0 / 5.0) * 4.0 * h, float(H_src)))
        H0 = int(clamp(H0_raw, H_min, H_max))
        W0 = int(min(int(H0 * target_aspect), W_src))

        x0 = x + 0.5 * w
        y0 = y + 0.5 * h
        x1 = int(x0 - 0.5 * W0)
        y1 = int(y0 - 0.75 * h - (1.0 / 6.0) * H0)

        x1 = clamp(x1, 0, max(0, W_src - W0))
        y1 = clamp(y1, 0, max(0, H_src - H0))

        slots.append({
            "slot_xywh": (dst_x, dst_y, dst_w, dst_h),
            "crop_xywh": (x1, y1, W0, H0),
            "track_id": d.get("track_id", None),
            "bbox": d["bbox"],
            "score": d.get("score", 0.0),
        })

    return {
        "rows": rows, "cols": cols,
        "cell_w": cell_w, "cell_h": cell_h,
        "slots": slots,
        "aspect_wh": aspect_wh,
        "target_face_h": targetFaceHeight,
        "out_w": out_w, "out_h": out_h,
    }


def compose_canvas(frame_bgr, plan):
    # Compose output canvas (handles pillarbox for single-person case)
    H_src, W_src = frame_bgr.shape[:2]
    out_w = plan.get("out_w", plan["cols"] * plan["cell_w"])
    out_h = plan.get("out_h", plan["rows"] * plan["cell_h"])
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    aspect_w, aspect_h = plan["aspect_wh"]
    target_aspect = aspect_w / aspect_h
    debug = {"intrusions": 0, "boundary_scores": []}

    for slot in plan["slots"]:
        sx, sy, sw, sh = slot["slot_xywh"]
        cx, cy, cw, ch = slot["crop_xywh"]

        x1 = max(0, cx)
        y1 = max(0, cy)
        x2 = min(W_src, cx + cw)
        y2 = min(H_src, cy + ch)
        roi = frame_bgr[y1:y2, x1:x2]

        if roi.size == 0:
            patch = np.zeros((sh, sw, 3), dtype=np.uint8)
        else:
            roi_h, roi_w = roi.shape[:2]
            roi_aspect = roi_w / max(roi_h, 1)
            if abs(roi_aspect - target_aspect) > 1e-3:
                if roi_aspect > target_aspect:
                    new_w = int(roi_h * target_aspect)
                    off = max((roi_w - new_w) // 2, 0)
                    roi = roi[:, off:off + new_w]
                else:
                    new_h = int(roi_w / target_aspect)
                    off = max((roi_h - new_h) // 2, 0)
                    roi = roi[off:off + new_h, :]
            patch = cv.resize(roi, (sw, sh), interpolation=cv.INTER_AREA)

        canvas[sy:sy + sh, sx:sx + sw] = patch

        bx, by, bw, bh = slot["bbox"]
        fcx = bx + bw / 2.0
        fcy = by + bh / 2.0
        left = (fcx - cx) / max(bh, 1)
        right = ((cx + cw) - fcx) / max(bh, 1)
        top = (fcy - cy) / max(bh, 1)
        bottom = ((cy + ch) - fcy) / max(bh, 1)
        boundary_score = min(left, right, top, bottom)
        debug["boundary_scores"].append(boundary_score)

    return canvas, debug
