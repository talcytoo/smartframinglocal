from __future__ import annotations

from typing import List, Dict, Tuple

import numpy as np


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1.0)


def _center(bbox):
    x, y, w, h = bbox
    return x + w * 0.5, y + h * 0.5


def _vert_iou(a, b):
    ay, ah = a[1], a[3]
    by, bh = b[1], b[3]
    lo = max(ay, by)
    hi = min(ay + ah, by + bh)
    inter = max(0, hi - lo)
    union = max(ah, 1) + max(bh, 1) - inter
    return inter / max(union, 1.0)


def deduplicate_overlap_faces(
    detections: List[Dict],
    stitch_dx: int,
    cam_width: int,
    overlap_start: int,
    overlap_end: int,
    iou_thr: float = 0.15,
    y_iou_thr: float = 0.3,
    max_center_dx: float = 80.0,
) -> List[Dict]:
    if len(detections) < 2 or overlap_end <= overlap_start:
        return detections

    ol = overlap_end - overlap_start
    if ol <= 0:
        return detections


    n = len(detections)
    remove = set()

    for i in range(n):
        if i in remove:
            continue
        bi = detections[i]["bbox"]
        ci = _center(bi)
        for j in range(i + 1, n):
            if j in remove:
                continue
            bj = detections[j]["bbox"]
            cj = _center(bj)

            hdist = abs(ci[0] - cj[0])
            if hdist > max_center_dx:
                continue

            if _vert_iou(bi, bj) < y_iou_thr:
                continue

            hi, hj = bi[3], bj[3]
            if hi <= 0 or hj <= 0:
                continue
            ratio = max(hi, hj) / max(min(hi, hj), 1)
            if ratio > 2.0:
                continue

            iou = _iou(bi, bj)
            if iou >= iou_thr or (hdist < max_center_dx * 0.6 and ratio < 1.5):
                si = detections[i].get("score", 0)
                sj = detections[j].get("score", 0)
                remove.add(j if si >= sj else i)

    return [d for idx, d in enumerate(detections) if idx not in remove]
