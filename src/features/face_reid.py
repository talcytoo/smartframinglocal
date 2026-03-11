from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import cv2 as cv

import mediapipe as mp


@dataclass
class ReIDConfig:
    model_asset_path: str = 'models/face_landmarker.task'
    max_faces: int = 1
    running_mode: str = 'IMAGE'
    reid_window_sec: float = 5.0
    sim_threshold: float = 0.18
    lmk_norm: bool = True
    use_blendshapes: bool = True


class FaceReID:

    def __init__(self, cfg: ReIDConfig):
        self.cfg = cfg
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

        mode = VisionRunningMode.IMAGE if cfg.running_mode.upper() == 'IMAGE' else VisionRunningMode.VIDEO
        self._options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=cfg.model_asset_path),
            running_mode=mode,
            num_faces=cfg.max_faces,
            output_face_blendshapes=cfg.use_blendshapes,
            output_facial_transformation_matrixes=False,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(self._options)

        self._gallery: list[tuple[int, float, np.ndarray]] = []
        self._last_embeds: dict[int, np.ndarray] = {}
        self._alias: dict[int, int] = {}

        self._last_debug: dict[int, dict] = {}
        self._last_match: dict[int, dict] = {}


    def clear(self):
        self._gallery.clear()
        self._last_embeds.clear()
        self._alias.clear()
        self._last_debug.clear()
        self._last_match.clear()

    def canonical_id(self, tid: int) -> int:
        visited = []
        cur = tid
        seen = set()
        while cur in self._alias:
            if cur in seen:
                self._alias.pop(cur, None)
                break
            seen.add(cur)
            visited.append(cur)
            cur = self._alias[cur]
        for v in visited:
            if v != cur:
                self._alias[v] = cur
        return cur

    def note_exit(self, tid: int):
        t_now = time.time()
        self._prune_gallery(t_now)
        emb = self._last_embeds.get(tid)
        if emb is not None:
            self._gallery.append((tid, t_now, emb))
            if len(self._gallery) > 16:
                self._gallery = self._gallery[-16:]

    def try_reassign(self, new_tid: int) -> int | None:
        t_now = time.time()
        self._prune_gallery(t_now)

        emb = self._last_embeds.get(new_tid)
        if emb is None or not self._gallery:
            return None

        best_old = None
        best_dist = 1e9
        for old_tid, t_exit, old_emb in self._gallery:
            d = cosine_distance(emb, old_emb)
            if d < best_dist:
                best_dist = d
                best_old = old_tid

        if best_old is None or best_dist > self.cfg.sim_threshold:
            return None

        old_canon = self.canonical_id(best_old)
        new_canon = self.canonical_id(new_tid)

        if new_canon == old_canon:
            self._gallery = [(tid, t, e) for (tid, t, e) in self._gallery if tid != best_old]
            self._last_match[new_tid] = {'old': old_canon, 'dist': best_dist, 't': t_now}
            return old_canon

        if self.canonical_id(old_canon) == new_tid:
            return None

        self._alias[new_tid] = old_canon
        _ = self.canonical_id(new_tid)

        self._gallery = [(tid, t, e) for (tid, t, e) in self._gallery if tid != best_old]
        self._last_match[new_tid] = {'old': old_canon, 'dist': best_dist, 't': t_now}
        return old_canon

    def update_track_embedding(self, tid: int, roi_bgr: np.ndarray):
        if roi_bgr is None or roi_bgr.size == 0:
            return

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv.cvtColor(roi_bgr, cv.COLOR_BGR2RGB)
        )
        res = self._landmarker.detect(mp_image)
        if not res.face_landmarks:
            return

        pts = np.array([[p.x, p.y, getattr(p, 'z', 0.0)] for p in res.face_landmarks[0]],
                       dtype=np.float32)

        if self.cfg.lmk_norm:
            xy = pts[:, :2]
            xy = (xy - xy.min(axis=0)) / np.clip((xy.max(axis=0) - xy.min(axis=0)), 1e-6, None)
            z = pts[:, 2:3]
            z = (z - z.mean()) / max(z.std(), 1e-6)
            pts_n = np.concatenate([xy, z], axis=1)
        else:
            pts_n = pts

        geom = geometry_feats_from_landmarks(pts_n)

        bs_scores = None
        bs_labels = None
        if self.cfg.use_blendshapes and res.face_blendshapes:
            cats = res.face_blendshapes[0]
            bs_scores = np.array([c.score for c in cats], dtype=np.float32)
            bs_labels = [c.category_name for c in cats]
            emb = np.concatenate([bs_scores, geom], axis=0)
        else:
            emb = geom

        emb = emb / max(np.linalg.norm(emb), 1e-6)
        self._last_embeds[tid] = emb

        self._last_debug[tid] = {
            'pts': pts_n,
            'bs_scores': bs_scores,
            'bs_labels': bs_labels,
        }

    def get_debug(self, tid: int) -> dict | None:
        return self._last_debug.get(tid)

    def get_last_match_info(self, tid: int) -> dict | None:
        return self._last_match.get(tid)


    def _prune_gallery(self, t_now: float):
        win = float(self.cfg.reid_window_sec)
        self._gallery = [(tid, t, e) for (tid, t, e) in self._gallery if (t_now - t) <= win]


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 1.0
    sim = float(np.dot(a, b) / denom)
    return 1.0 - sim


def geometry_feats_from_landmarks(pts: np.ndarray) -> np.ndarray:
    idx = {
        'lip_up': 13, 'lip_lo': 14,
        'le_up': 159, 'le_lo': 145,
        're_up': 386, 're_lo': 374,
        'nose': 1, 'le_outer': 33, 're_outer': 263,
    }

    def safe(i: int):
        i = max(0, min(int(i), pts.shape[0] - 1))
        return pts[i]

    lu = safe(idx['lip_up']); ll = safe(idx['lip_lo'])
    leu = safe(idx['le_up']);  lel = safe(idx['le_lo'])
    reu = safe(idx['re_up']);  rel = safe(idx['re_lo'])
    nose = safe(idx['nose'])
    le_outer = safe(idx['le_outer']); re_outer = safe(idx['re_outer'])

    mouth_gap = np.linalg.norm(lu[:2] - ll[:2])
    le_aspect = np.linalg.norm(leu[:2] - lel[:2])
    re_aspect = np.linalg.norm(reu[:2] - rel[:2])
    eye_span = np.linalg.norm(le_outer[:2] - re_outer[:2]) + 1e-6
    eyes_center = 0.5 * (le_outer[:2] + re_outer[:2])
    nose_eyes = np.linalg.norm(nose[:2] - eyes_center)

    v = np.array([
        mouth_gap / eye_span,
        le_aspect / eye_span,
        re_aspect / eye_span,
        nose_eyes / eye_span,
        pts[:, 2].mean(),
        pts[:, 2].std(),
    ], dtype=np.float32)

    return v
