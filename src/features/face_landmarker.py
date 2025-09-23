import cv2 as cv
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe import Image, ImageFormat


@dataclass
class SpeakingState:
    # Tracks EMA score and speaking state for one track_id
    ema: float = 0.0
    is_speaking: bool = False
    last_ts: float = 0.0
    cooloff: int = 0  # sustain counter after speaking drops


class FaceLandmarkerASD:
    # Wrapper around MediaPipe Face Landmarker for simple active-speaker detection
    # Outputs lip-open score [0,1] and debounced speaking boolean per track_id
    def __init__(
        self,
        model_asset_path: Optional[str] = None,   # None → use default bundled model
        running_mode: str = "VIDEO",             # IMAGE | VIDEO | LIVE_STREAM
        max_faces: int = 1,                      # expect 1 face per ROI
        use_blendshapes: bool = True,
        ema_alpha: float = 0.55,                 # EMA smoothing factor
        on_thr: float = 0.35,                    # threshold to switch ON speaking
        off_thr: float = 0.22,                   # lower threshold to switch OFF
        sustain_frames: int = 6,                 # sustain speaking N frames after drop
        lips_fallback: bool = True               # fallback to lip-gap if no blendshapes
    ):
        BaseOptions = mp_python.BaseOptions
        FaceLandmarker = mp_vision.FaceLandmarker
        FaceLandmarkerOptions = mp_vision.FaceLandmarkerOptions
        RunningMode = mp_vision.RunningMode

        self._states: Dict[int, SpeakingState] = {}
        self._ema_alpha = float(ema_alpha)
        self._on_thr = float(on_thr)
        self._off_thr = float(off_thr)
        self._sustain = int(sustain_frames)
        self._use_blend = bool(use_blendshapes)
        self._fallback = bool(lips_fallback)

        rm = {
            "IMAGE": RunningMode.IMAGE,
            "VIDEO": RunningMode.VIDEO,
            "LIVE_STREAM": RunningMode.LIVE_STREAM,
        }[running_mode]

        opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path) if model_asset_path else BaseOptions(),
            running_mode=rm,
            num_faces=max_faces,
            output_face_blendshapes=self._use_blend,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(opts)

    def _blendshape_jaw_open(self, blendshapes) -> Optional[float]:
        # Extract jaw/mouth open score from blendshapes
        if not blendshapes:
            return None
        bs = blendshapes[0]
        for c in bs:
            if c.category_name in ("jawOpen", "mouthOpen"):
                return float(c.score)
        return None

    def _lip_gap_score(self, landmarks) -> Optional[float]:
        # Compute normalized lip-gap score using landmarks
        if not landmarks:
            return None
        lm = landmarks[0]
        try:
            def xy(i):
                p = lm[i]
                return np.array([p.x, p.y], dtype=np.float32)

            upper = xy(13)
            lower = xy(14)
            r_eye = xy(33)
            l_eye = xy(263)

            gap = np.linalg.norm(upper - lower)
            eye = np.linalg.norm(r_eye - l_eye)
            if eye <= 1e-6:
                return None

            raw = gap / (eye * 0.45)  # empirical eye-to-mouth ratio
            return float(np.clip(raw, 0.0, 1.0))
        except Exception:
            return None

    def score_and_update(
        self,
        bgr_roi: np.ndarray,
        track_id: int,
        frame_timestamp_ms: Optional[int] = None
    ) -> Tuple[float, bool]:
        # Returns (lip_open_score in [0,1], is_speaking boolean)
        if bgr_roi is None or bgr_roi.size == 0:
            st = self._states.setdefault(track_id, SpeakingState())
            return st.ema, st.is_speaking

        # Convert ROI to MediaPipe image
        rgb = cv.cvtColor(bgr_roi, cv.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        # Get score from blendshapes or fallback
        score = None
        if self._use_blend:
            score = self._blendshape_jaw_open(getattr(result, "face_blendshapes", None))
        if score is None and self._fallback:
            score = self._lip_gap_score(getattr(result, "face_landmarks", None))
        if score is None:
            score = 0.0

        # Update state for this track
        st = self._states.setdefault(track_id, SpeakingState())
        st.ema = (1.0 - self._ema_alpha) * st.ema + self._ema_alpha * score

        if st.is_speaking:
            # Drop only when below off_thr, with sustain
            if st.ema < self._off_thr:
                if st.cooloff > 0:
                    st.cooloff -= 1
                else:
                    st.is_speaking = False
            else:
                st.cooloff = self._sustain
        else:
            # Activate when above on_thr
            if st.ema > self._on_thr:
                st.is_speaking = True
                st.cooloff = self._sustain

        return st.ema, st.is_speaking
