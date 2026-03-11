from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class DualCamConfig:
    cam0_index: int = 0
    cam1_index: int = 1
    width: int = 1280
    height: int = 720
    fps: int = 30

    max_features: int = 1500
    ratio: float = 0.75
    min_agree: int = 20

    windowed: bool = False
    window_frac: float = 0.5

    smooth_alpha: float = 0.2

    auto_seam: bool = True
    seam_frac: float = 0.5
    blend_width: int = 10

    color_correct: bool = True
    cc_strength: float = 0.8

    recalc_every: int = 1
    cc_refresh_mult: int = 30


class _FrameGrabber:
    __slots__ = ('cap', '_lock', '_frame', '_stopped', '_th')

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stopped = False
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _loop(self):
        while not self._stopped:
            ok, frm = self.cap.read()
            if ok and frm is not None:
                with self._lock:
                    self._frame = frm
            time.sleep(0.0005)

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self):
        self._stopped = True
        self._th.join(timeout=1.0)


def _open_camera(index: int, w: int, h: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def _resize_common(a: np.ndarray, b: np.ndarray):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return (cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA),
            cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA))


def _estimate_shift_full(img0, img1, max_features=1500, ratio=0.75):
    g0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=10)
    k0, d0 = orb.detectAndCompute(g0, None)
    k1, d1 = orb.detectAndCompute(g1, None)
    if d0 is None or d1 is None or len(k0) < 12 or len(k1) < 12:
        return None, 0
    knn = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False).knnMatch(d0, d1, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < ratio * n.distance]
    if len(good) < 12:
        return None, 0
    dxs = np.array([k0[m.queryIdx].pt[0] - k1[m.trainIdx].pt[0] for m in good])
    dx = int(np.median(dxs))
    agree = int(np.sum(np.abs(dxs - dx) <= 6.0))
    return dx, agree


def _estimate_shift_windowed(img0, img1, window_frac, max_features=1500, ratio=0.75):
    w0, w1 = img0.shape[1], img1.shape[1]
    frac = float(np.clip(window_frac, 0.1, 1.0))
    keep_l = max(1, min(int(round(frac * w0)), w0))
    keep_r = max(1, min(int(round(frac * w1)), w1))
    off0 = w0 - keep_l
    g0 = cv2.cvtColor(img0[:, off0:], cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(img1[:, :keep_r], cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=10)
    k0, d0 = orb.detectAndCompute(g0, None)
    k1, d1 = orb.detectAndCompute(g1, None)
    if d0 is None or d1 is None or len(k0) < 12 or len(k1) < 12:
        return None, 0
    knn = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False).knnMatch(d0, d1, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < ratio * n.distance]
    if len(good) < 12:
        return None, 0
    dxs = np.array([k0[m.queryIdx].pt[0] - k1[m.trainIdx].pt[0] for m in good])
    med = float(np.median(dxs))
    dx = int(med) + off0
    agree = int(np.sum(np.abs(dxs - med) <= 6.0))
    return dx, agree


def _compute_cc(img0, img1, dx, strength=0.8):
    w0, w1 = img0.shape[1], img1.shape[1]
    os_ = max(0, -dx)
    oe = min(w1, w0 - dx)
    ol = oe - os_
    if ol < 20:
        return np.ones(3, np.float32), np.zeros(3, np.float32)
    margin = ol // 4
    ss, se = os_ + margin, oe - margin
    sw = se - ss
    if sw < 10:
        return np.ones(3, np.float32), np.zeros(3, np.float32)
    rx0 = int(np.clip(ss + dx, 0, w0 - 1))
    rw0 = min(sw, w0 - rx0)
    rx1 = int(np.clip(ss, 0, w1 - 1))
    rw1 = min(sw, w1 - rx1)
    if rw0 < 10 or rw1 < 10:
        return np.ones(3, np.float32), np.zeros(3, np.float32)
    cw = min(rw0, rw1)
    p0 = img0[:, rx0:rx0 + cw]
    p1 = img1[:, rx1:rx1 + cw]
    gain = np.ones(3, np.float32)
    off = np.zeros(3, np.float32)
    for c in range(3):
        m0, m1 = float(p0[:, :, c].mean()), float(p1[:, :, c].mean())
        s0, s1 = float(p0[:, :, c].std()), float(p1[:, :, c].std())
        g = np.clip(s0 / s1, 0.5, 2.0) if s1 > 1 and s0 > 1 else 1.0
        o = np.clip(m0 - g * m1, -50, 50)
        gain[c] = 1.0 + strength * (g - 1.0)
        off[c] = strength * o
    return gain, off


def _apply_cc(src, gain, off):
    dst = src.astype(np.float32)
    for c in range(3):
        dst[:, :, c] = dst[:, :, c] * gain[c] + off[c]
    return np.clip(dst, 0, 255).astype(np.uint8)


def _optimal_seam_frac(dx, w0, w1, margin=0.05):
    os_ = max(0, -dx)
    oe = min(w1, w0 - dx)
    ol = oe - os_
    if ol <= 0:
        return 0.5
    ow = w0 + w1 - ol
    sx0 = ow // 2
    sx1 = sx0 - dx
    frac = (sx1 - os_) / float(ol)
    return float(np.clip(frac, margin, 1.0 - margin))


def _blend_join(img0, img1, dx, seam_frac=0.5, blend_width=50):
    h, w0 = img0.shape[:2]
    w1 = img1.shape[1]
    os_ = int(np.clip(max(0, -dx), 0, w1))
    oe = int(np.clip(min(w1, w0 - dx), os_, w1))
    ol = oe - os_
    if ol <= 0:
        return np.hstack([img0, img1]), w0, 0, 0

    sf = float(np.clip(seam_frac, 0.0, 1.0))
    sx1 = int(np.clip(os_ + round(sf * ol), 1, w1 - 1))
    sx0 = int(np.clip(sx1 + dx, 1, w0 - 1))

    hb = blend_width // 2
    bs0 = max(1, sx0 - hb)
    be0 = min(w0 - 1, sx0 + hb)
    ab = be0 - bs0
    if ab < 4:
        return np.hstack([img0[:, :sx0], img1[:, sx1:]]), sx0, os_, oe

    left = img0[:, :bs0]
    bs1 = int(np.clip(bs0 - dx, 0, w1 - 1))
    be1 = int(np.clip(be0 - dx, bs1 + 1, w1))
    bl1 = be1 - bs1

    b0 = img0[:, bs0:bs0 + ab]
    if bl1 > 0 and bl1 == ab:
        b1 = img1[:, bs1:bs1 + bl1]
    else:
        b1 = cv2.resize(img1[:, bs1:bs1 + max(1, bl1)], (ab, h))

    alpha = np.linspace(0, 1, ab, dtype=np.float32)[np.newaxis, :, np.newaxis]
    blended = np.clip((1 - alpha) * b0.astype(np.float32) + alpha * b1.astype(np.float32),
                      0, 255).astype(np.uint8)

    rs1 = int(np.clip(be1, 0, w1 - 1))
    right = img1[:, rs1:]
    out = np.hstack([left, blended, right])
    return out, bs0 + ab // 2, os_, oe


class DualCamStitcher:

    def __init__(self, cfg: DualCamConfig):
        self.cfg = cfg
        self._cap0 = self._cap1 = None
        self._g0 = self._g1 = None
        self._started = False

        self.dx: int | None = None
        self.has_dx = False
        self._smoothed_dx = 0.0
        self._seam_frac = cfg.seam_frac
        self._cc_gain = np.ones(3, np.float32)
        self._cc_off = np.zeros(3, np.float32)

        self.overlap_range = (0, 0)
        self.cam_width = 0

        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._raw0: np.ndarray | None = None
        self._raw1: np.ndarray | None = None
        self._frame_count = 0
        self._stitch_thread: threading.Thread | None = None
        self._stop_flag = False


    def start(self):
        if self._started:
            return
        c = self.cfg
        self._cap0 = _open_camera(c.cam0_index, c.width, c.height, c.fps)
        self._cap1 = _open_camera(c.cam1_index, c.width, c.height, c.fps)
        self._g0 = _FrameGrabber(self._cap0)
        self._g1 = _FrameGrabber(self._cap1)
        for _ in range(20):
            self._g0.read()
            self._g1.read()
            time.sleep(0.01)
        self._started = True
        self._stop_flag = False
        self._stitch_thread = threading.Thread(target=self._stitch_loop, daemon=True)
        self._stitch_thread.start()

    def stop(self):
        self._stop_flag = True
        if self._stitch_thread:
            self._stitch_thread.join(timeout=2.0)
        if self._g0:
            self._g0.stop()
        if self._g1:
            self._g1.stop()
        if self._cap0:
            self._cap0.release()
        if self._cap1:
            self._cap1.release()
        self._started = False

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def read_raw(self):
        with self._lock:
            if self._raw0 is None or self._raw1 is None:
                return False, None, None
            return True, self._raw0.copy(), self._raw1.copy()


    def _stitch_loop(self):
        c = self.cfg
        ok0, f0 = self._g0.read()
        ok1, f1 = self._g1.read()
        if ok0 and ok1:
            f0, f1 = _resize_common(f0, f1)
            self.cam_width = f0.shape[1]
            dx, ag = self._run_shift(f0, f1)
            if dx is not None and ag >= c.min_agree:
                self.dx = dx
                self.has_dx = True
                self._smoothed_dx = float(dx)
                if c.color_correct:
                    self._cc_gain, self._cc_off = _compute_cc(f0, f1, dx, c.cc_strength)

        frame_idx = 0
        while not self._stop_flag:
            ok0, f0 = self._g0.read()
            ok1, f1 = self._g1.read()
            if not ok0 or not ok1:
                time.sleep(0.002)
                continue

            f0, f1 = _resize_common(f0, f1)
            w = f0.shape[1]
            self.cam_width = w
            frame_idx += 1

            if c.recalc_every and (frame_idx % c.recalc_every == 0):
                dx_try, ag_try = self._run_shift(f0, f1)
                if dx_try is not None and ag_try >= c.min_agree:
                    self._smoothed_dx = (c.smooth_alpha * dx_try +
                                         (1 - c.smooth_alpha) * self._smoothed_dx)
                    self.dx = int(round(self._smoothed_dx))
                    self.has_dx = True
                if (self.has_dx and c.color_correct and c.recalc_every and
                        frame_idx % (c.recalc_every * c.cc_refresh_mult) == 0):
                    self._cc_gain, self._cc_off = _compute_cc(f0, f1, self.dx, c.cc_strength)

            sf = self._seam_frac
            if c.auto_seam and self.has_dx:
                sf = _optimal_seam_frac(self.dx, w, w)
                self._seam_frac = sf

            if self.has_dx:
                if c.color_correct:
                    f1c = _apply_cc(f1, self._cc_gain, self._cc_off)
                else:
                    f1c = f1
                out, _, os_, oe = _blend_join(f0, f1c, self.dx, sf, c.blend_width)
                self.overlap_range = (os_, oe)
            else:
                out = np.hstack([f0, f1])
                self.overlap_range = (0, 0)

            with self._lock:
                self._frame = out
                self._raw0 = f0
                self._raw1 = f1
                self._frame_count = frame_idx

    def _run_shift(self, f0, f1):
        c = self.cfg
        if c.windowed:
            return _estimate_shift_windowed(f0, f1, c.window_frac, c.max_features, c.ratio)
        else:
            return _estimate_shift_full(f0, f1, c.max_features, c.ratio)
