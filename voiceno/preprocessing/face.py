"""Face landmark detection using MediaPipe FaceLandmarker.

Detects facial keypoints (eyes, nose, mouth center) across video frames
and provides temporal interpolation for missed detections.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceDetector:
    """Detects facial landmarks across video frames using MediaPipe FaceLandmarker."""

    def __init__(self, model_asset_path: Optional[Union[str, Path]] = None):
        if model_asset_path is None:
            model_asset_path = Path(__file__).resolve().parent / "face_landmarker.task"

        self.model_path = str(model_asset_path)
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect key landmarks for a single RGB frame.

        Returns:
            np.ndarray of shape (4, 2) containing [right_eye, left_eye, nose_tip, mouth_center]
            in pixel coordinates, or None if no face is detected.
        """
        ih, iw, _ = frame.shape
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)

        # Ensure contiguous RGB
        frame_rgb = np.ascontiguousarray(frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return None

        lm = detection_result.face_landmarks[0]

        # Key landmark points:
        # 0: Right eye center (mean of indices 33, 133, 159, 145)
        # 1: Left eye center (mean of indices 263, 362, 386, 374)
        # 2: Nose tip (landmark 4)
        # 3: Mouth center (mean of outer and inner lip contour landmarks 61, 291, 0, 17, 13, 14)
        r_eye = np.mean([[lm[i].x * iw, lm[i].y * ih] for i in [33, 133, 159, 145]], axis=0)
        l_eye = np.mean([[lm[i].x * iw, lm[i].y * ih] for i in [263, 362, 386, 374]], axis=0)
        nose = np.array([lm[4].x * iw, lm[4].y * ih])
        mouth = np.mean([[lm[i].x * iw, lm[i].y * ih] for i in [61, 291, 0, 17, 13, 14]], axis=0)

        landmarks = np.array([r_eye, l_eye, nose, mouth], dtype=np.float32)
        return landmarks

    def __call__(self, video_frames: Union[np.ndarray, List[np.ndarray]]) -> List[Optional[np.ndarray]]:
        """Detect landmarks for each frame in video_frames."""
        return [self.detect_frame(f) for f in video_frames]

    @staticmethod
    def interpolate_landmarks(landmarks: List[Optional[np.ndarray]]) -> Optional[List[np.ndarray]]:
        """Linearly interpolate missing landmark frames in a video sequence.

        Handles leading and trailing missing frames by edge replication.
        """
        valid_indices = [i for i, lm in enumerate(landmarks) if lm is not None]
        if not valid_indices:
            return None

        interpolated = [lm.copy() if lm is not None else None for lm in landmarks]

        # Interpolate between valid points
        for i in range(1, len(valid_indices)):
            prev_idx = valid_indices[i - 1]
            next_idx = valid_indices[i]
            if next_idx - prev_idx > 1:
                start_pts = interpolated[prev_idx]
                end_pts = interpolated[next_idx]
                delta = end_pts - start_pts
                span = float(next_idx - prev_idx)
                for step in range(1, next_idx - prev_idx):
                    interpolated[prev_idx + step] = start_pts + (step / span) * delta

        # Fill leading missing frames
        first_valid = valid_indices[0]
        for i in range(first_valid):
            interpolated[i] = interpolated[first_valid].copy()

        # Fill trailing missing frames
        last_valid = valid_indices[-1]
        for i in range(last_valid + 1, len(interpolated)):
            interpolated[i] = interpolated[last_valid].copy()

        return interpolated

    def close(self):
        """Releases detector resources."""
        if hasattr(self, "landmarker") and self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass

    def __del__(self):
        self.close()
