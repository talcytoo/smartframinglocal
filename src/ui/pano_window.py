# src/ui/pano_window.py
from dataclasses import dataclass
import time
import cv2 as cv
import numpy as np


@dataclass
class PanoConfig:
    enabled: bool = True
    width: int = 480
    height: int = 270
    collapsed_h: int = 18
    margin_bottom: int = 24
    expand_ms: int = 220
    collapse_ms: int = 160
    hover_radius: int = 72
    auto_hide_sec: float = 2.5       # hold time after mouse leaves (sec)
    boot_intro_sec: float = 10.0     # boot intro duration before collapsing
    border_px: int = 2
    radius_px: int = 10
    bg_color: tuple = (24, 24, 24)       # panel background (BGR)
    border_color: tuple = (64, 64, 64)   # border color (BGR)
    arrow_color: tuple = (180, 180, 180) # arrow color (BGR)
    shadow: bool = True


class PanoWindow:
    # Collapsible panoramic mini-window showing full camera FOV.

    def __init__(self, canvas_w: int, canvas_h: int, cfg: PanoConfig):
        self.cfg = cfg
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h

        # Finite state machine states: collapsed | expanding | expanded | collapsing
        self.state = "collapsed"
        self._t0 = time.time()
        self._p = 0.0

        # Mouse and timing state
        self.mouse_xy = (self.canvas_w // 2, self.canvas_h - 1)
        self._last_mouse_in = False
        self._last_leave_time = time.time()
        self._last_user_expand_time = None

        # Boot intro control
        self._boot_start_time = time.time()
        self._boot_intro_active = True
        self._meeting_started = False

        # Layout
        self._recompute_layouts()

        # Start boot intro expansion if enabled
        if self.cfg.enabled and self.cfg.boot_intro_sec > 0.0:
            self.state = "expanding"
            self._t0 = time.time()

    @staticmethod
    def draw_shadow(img, rect, offset=(4, 4), alpha=0.25):
        x, y, w, h = rect
        ox, oy = offset
        shadow = img.copy()
        cv.rectangle(shadow, (x + ox, y + oy), (x + w + ox, y + h + oy), (0, 0, 0), -1)
        cv.addWeighted(shadow, alpha, img, 1 - alpha, 0, img)

    @staticmethod
    def fill_round_rect(img, rect, radius, color):
        x, y, w, h = rect
        overlay = img.copy()
        cv.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), color, -1)
        cv.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), color, -1)
        cv.circle(overlay, (x + radius, y + radius), radius, color, -1)
        cv.circle(overlay, (x + w - radius, y + radius), radius, color, -1)
        cv.circle(overlay, (x + radius, y + h - radius), radius, color, -1)
        cv.circle(overlay, (x + w - radius, y + h - radius), radius, color, -1)
        cv.addWeighted(overlay, 1.0, img, 0.0, 0, img)

    @staticmethod
    def stroke_round_rect(img, rect, radius, color, thickness=2):
        x, y, w, h = rect
        cv.rectangle(img, (x + radius, y), (x + w - radius, y + h), color, thickness)
        cv.rectangle(img, (x, y + radius), (x + w, y + h - radius), color, thickness)
        cv.ellipse(img, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv.ellipse(img, (x + w - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv.ellipse(img, (x + radius, y + h - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv.ellipse(img, (x + w - radius, y + h - radius), (radius, radius), 0, 0, 90, color, thickness)

    def _recompute_layouts(self):
        c = self.cfg
        cx = self.canvas_w // 2
        w = min(c.width, self.canvas_w - 40)
        h = min(c.height, self.canvas_h // 3)
        h = max(h, c.collapsed_h)
        x0 = cx - w // 2
        y1 = self.canvas_h - c.margin_bottom
        y0_expanded = y1 - h
        y0_collapsed = y1 - c.collapsed_h
        self.rect_expanded = (x0, y0_expanded, w, h)
        self.rect_collapsed = (x0, y0_collapsed, w, c.collapsed_h)
        self.arrow_center = (cx, y1 - c.collapsed_h // 2)

    def set_canvas_size(self, w: int, h: int):
        if (w, h) != (self.canvas_w, self.canvas_h):
            self.canvas_w, self.canvas_h = w, h
            self._recompute_layouts()

    def set_mouse(self, x: int, y: int):
        self.mouse_xy = (x, y)

    def notify_activity(self, any_speaking: bool):
        # If someone speaks during boot intro, mark meeting started and collapse immediately
        if any_speaking and not self._meeting_started:
            self._meeting_started = True
            self._boot_intro_active = False
            if self.state in ("expanding", "expanded"):
                self.state = "collapsing"
                self._t0 = time.time()

    def _in_proximity(self) -> bool:
        x, y = self.mouse_xy
        ax, ay = self.arrow_center
        dx, dy = x - ax, y - ay
        return (dx * dx + dy * dy) <= (self.cfg.hover_radius ** 2)

    def _update_state(self):
        now = time.time()

        # Boot intro behavior
        if self._boot_intro_active and not self._meeting_started:
            elapsed = now - self._boot_start_time
            if self.state in ("expanding", "expanded"):
                if self.state == "expanding":
                    t = min(1.0, (now - self._t0) * 1000.0 / self.cfg.expand_ms)
                    self._p = t
                    if t >= 1.0:
                        self.state = "expanded"
                else:
                    self._p = 1.0
                if elapsed >= self.cfg.boot_intro_sec:
                    self.state = "collapsing"
                    self._t0 = now
                    self._boot_intro_active = False
            elif self.state == "collapsing":
                t = min(1.0, (now - self._t0) * 1000.0 / self.cfg.collapse_ms)
                self._p = 1.0 - t
                if t >= 1.0:
                    self.state = "collapsed"
                    self._boot_intro_active = False
            elif self.state == "collapsed":
                if elapsed < self.cfg.boot_intro_sec:
                    self.state = "expanding"
                    self._t0 = now
                    self._p = 0.0
                else:
                    self._boot_intro_active = False
            return

        # Mouse-driven behavior
        near = self._in_proximity()
        if near:
            if self.state in ("collapsed", "collapsing"):
                self.state = "expanding"
                self._t0 = now
                self._last_user_expand_time = now
            self._last_mouse_in = True
        else:
            if self._last_mouse_in:
                self._last_mouse_in = False
                self._last_leave_time = now

            hold_until = None
            if self._last_user_expand_time is not None:
                hold_until = self._last_user_expand_time + float(self.cfg.auto_hide_sec)

            leave_ok = (now - self._last_leave_time) >= float(self.cfg.auto_hide_sec)
            hold_ok = (hold_until is None) or (now >= hold_until)

            if leave_ok and hold_ok and self.state in ("expanded", "expanding"):
                self.state = "collapsing"
                self._t0 = now

        # Advance animation
        if self.state == "expanding":
            t = min(1.0, (now - self._t0) * 1000.0 / self.cfg.expand_ms)
            self._p = t
            if t >= 1.0:
                self.state = "expanded"
        elif self.state == "collapsing":
            t = min(1.0, (now - self._t0) * 1000.0 / self.cfg.collapse_ms)
            self._p = 1.0 - t
            if t >= 1.0:
                self.state = "collapsed"
                self._last_user_expand_time = None
        elif self.state == "collapsed":
            self._p = 0.0
        elif self.state == "expanded":
            self._p = 1.0

    @staticmethod
    def lerp_rect(r0, r1, p):
        x0, y0, w0, h0 = r0
        x1, y1, w1, h1 = r1
        x = int(x0 + (x1 - x0) * p)
        y = int(y0 + (y1 - y0) * p)
        w = int(w0 + (w1 - w0) * p)
        h = int(h0 + (h1 - h0) * p)
        return x, y, w, h

    def update_and_draw(self, canvas_bgr, full_frame_bgr):
        if not self.cfg.enabled:
            return canvas_bgr

        # Update FSM and ease animation
        self._update_state()
        p = self._p
        p = p * p * (3 - 2 * p)  # smoothstep easing
        rect = self.lerp_rect(self.rect_collapsed, self.rect_expanded, p)
        x, y, w, h = rect

        # Draw panel
        if self.cfg.shadow:
            self.draw_shadow(canvas_bgr, rect)
        self.fill_round_rect(canvas_bgr, rect, self.cfg.radius_px, self.cfg.bg_color)
        self.stroke_round_rect(canvas_bgr, rect, self.cfg.radius_px, self.cfg.border_color, self.cfg.border_px)

        # Arrow (up if collapsed-ish, down if expanded-ish)
        arrow_h = self.cfg.collapsed_h
        ay = y + h - arrow_h // 2
        cx = x + w // 2
        pts = np.array([
            [cx - 10, ay + (3 if self._p < 0.5 else -3)],
            [cx + 10, ay + (3 if self._p < 0.5 else -3)],
            [cx, ay + (-6 if self._p < 0.5 else 6)],
        ], dtype=np.int32)
        cv.fillConvexPoly(canvas_bgr, pts, self.cfg.arrow_color)

        # Mini video preview
        if h > arrow_h + 12:
            pad = 8
            inner_w, inner_h = w - 2 * pad, h - arrow_h - 2 * pad
            if inner_w > 4 and inner_h > 4:
                fh, fw = full_frame_bgr.shape[:2]
                scale = min(inner_w / fw, inner_h / fh)
                rw, rh = max(1, int(fw * scale)), max(1, int(fh * scale))
                mini = cv.resize(full_frame_bgr, (rw, rh), interpolation=cv.INTER_AREA)
                x0 = x + (w - rw) // 2
                y0 = y + pad
                canvas_bgr[y0:y0 + rh, x0:x0 + rw] = mini

        return canvas_bgr
