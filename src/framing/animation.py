import time
from typing import Dict, Any, Optional


class LayoutAnimator:
    # Smooths layout changes using exponential decay.
    # Helps reduce jitter in slot assignments across frames.
    def __init__(self, tau: float = 0.15, shrink_in: float = 0.85):
        self.tau = tau              # decay constant for interpolation
        self.shrink_in = shrink_in  # factor applied when a slot first appears
        self.last_time: Optional[float] = None
        self.prev_plan: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        # Clear animator state
        self.last_time = None
        self.prev_plan = None

    def update(self, plan: Dict[str, Any], dt: Optional[float] = None) -> Dict[str, Any]:
        # Update layout with smoothing between frames
        now = time.time()
        if dt is None:
            if self.last_time is None:
                dt = 0.0
            else:
                dt = now - self.last_time
        self.last_time = now

        if self.prev_plan is None:
            # First call: no smoothing
            self.prev_plan = plan
            return plan

        # Exponential interpolation factor
        alpha = 1.0 - pow(2.71828, -dt / max(1e-6, self.tau))

        prev_slots = {
            s["track_id"]: s
            for s in self.prev_plan.get("slots", [])
            if s.get("track_id") is not None
        }
        new_slots = []
        for slot in plan.get("slots", []):
            tid = slot.get("track_id")
            if tid is not None and tid in prev_slots:
                prev = prev_slots[tid]["slot_xywh"]
                cur = slot["slot_xywh"]
                smoothed = tuple(
                    int((1 - alpha) * p + alpha * c) for p, c in zip(prev, cur)
                )
                slot["slot_xywh"] = smoothed
            new_slots.append(slot)

        smoothed_plan = dict(plan)
        smoothed_plan["slots"] = new_slots
        self.prev_plan = smoothed_plan
        return smoothed_plan
