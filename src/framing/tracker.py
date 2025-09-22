from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import itertools
import math
import time

BBox = Tuple[int, int, int, int]


def _iou(a: BBox, b: BBox) -> float:
    # Intersection-over-Union for two bounding boxes
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)

    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / max(1.0, union)


def _center_dist(a: BBox, b: BBox) -> float:
    # Euclidean distance between centers of two boxes
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    acx, acy = ax + aw * 0.5, ay + ah * 0.5
    bcx, bcy = bx + bw * 0.5, by + bh * 0.5
    return math.hypot(acx - bcx, acy - bcy)


@dataclass
class Track:
    tid: int
    bbox: BBox
    hits: int = 0
    misses: int = 0
    last_seen_t: float = 0.0
    confirmed: bool = False


class StableTracker:
    # IoU-based tracker with optional center-distance gating
    # Tracks require N hits before confirmation; removed when miss/forget thresholds are exceeded
    def __init__(
        self,
        match_iou_thr: float = 0.3,
        forget_after: int = 30,
        enter_confirm: int = 2,
        exit_confirm: int = 8,
        center_dist_px: Optional[int] = None,
    ):
        self.match_iou_thr = match_iou_thr
        self.forget_after = forget_after
        self.enter_confirm = enter_confirm
        self.exit_confirm = exit_confirm

        try:
            from .. import config
            self.center_dist_px = (
                center_dist_px
                if center_dist_px is not None
                else getattr(config, "CENTER_DIST_PX", None)
            )
        except Exception:
            self.center_dist_px = center_dist_px

        self.tracks: Dict[int, Track] = {}
        self.next_id = itertools.count(1)
        self.frame_idx = 0

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = itertools.count(1)
        self.frame_idx = 0

    def _assign_active(self, dets: List[BBox]) -> List[Optional[int]]:
        tids = list(self.tracks.keys())
        candidates = []
        for di, db in enumerate(dets):
            for tid in tids:
                tr = self.tracks[tid]
                iou = _iou(tr.bbox, db)
                if iou < self.match_iou_thr:
                    continue
                if (
                    self.center_dist_px is not None
                    and _center_dist(tr.bbox, db) > self.center_dist_px
                ):
                    continue
                candidates.append((iou, -_center_dist(tr.bbox, db), tid, di))

        candidates.sort(reverse=True)
        used_t, used_d = set(), set()
        out: List[Optional[int]] = [None] * len(dets)

        for iou, neg_cd, tid, di in candidates:
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid)
            used_d.add(di)
            out[di] = tid
        return out

    def update(self, dets: List[BBox], frame_bgr: Optional[object] = None) -> List[Optional[int]]:
        # Update tracker with detections. Returns list of track IDs aligned with dets.
        self.frame_idx += 1
        now = time.time()

        # 1. Age all active tracks
        for tr in self.tracks.values():
            tr.misses += 1

        # 2. Assign detections to existing tracks
        assigned = self._assign_active(dets)

        # 3. Update matched tracks
        for di, db in enumerate(dets):
            tid = assigned[di]
            if tid is None or tid not in self.tracks:
                continue
            tr = self.tracks[tid]
            tr.bbox = db
            tr.hits += 1
            tr.misses = 0
            tr.last_seen_t = now
            if not tr.confirmed and tr.hits >= self.enter_confirm:
                tr.confirmed = True

        # 4. Remove tracks that exceeded miss/forget thresholds
        to_forget = []
        for tid, tr in self.tracks.items():
            miss_limit = self.exit_confirm if tr.confirmed else max(2, self.enter_confirm + 1)
            if tr.misses >= miss_limit or tr.misses >= self.forget_after:
                to_forget.append(tid)
        for tid in to_forget:
            self.tracks.pop(tid, None)

        # 5. Create new tracks for unmatched detections
        for di, db in enumerate(dets):
            if assigned[di] is None:
                tid = next(self.next_id)
                tr = Track(
                    tid=tid,
                    bbox=db,
                    hits=1,
                    misses=0,
                    last_seen_t=now,
                    confirmed=(self.enter_confirm <= 1),
                )
                self.tracks[tid] = tr
                assigned[di] = tid

        return assigned