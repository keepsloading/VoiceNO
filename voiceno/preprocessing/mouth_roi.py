"""Mouth ROI extraction with affine face alignment and temporal smoothing.

Extracts normalized 96x96 mouth patches aligned with standard reference landmarks.
Also supports representation ablations (mouth-only, mouth+jaw, full face).
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np


class MouthROIExtractor:
    """Extracts aligned mouth region-of-interest patches from video frames."""

    def __init__(
        self,
        mean_face_path: Optional[Union[str, Path]] = None,
        crop_width: int = 96,
        crop_height: int = 96,
        window_margin: int = 12,
        representation: str = "mouth",
    ):
        if mean_face_path is None:
            mean_face_path = Path(__file__).resolve().parent / "20words_mean_face.npy"

        self.reference = np.load(str(mean_face_path))
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.window_margin = window_margin
        self.representation = representation

    def get_stable_reference(
        self,
        target_size: Tuple[int, int] = (256, 256),
        reference_size: Tuple[int, int] = (256, 256),
    ) -> np.ndarray:
        """Computes reference points for right eye, left eye, nose tip, and mouth center."""
        stable_ref = np.vstack([
            np.mean(self.reference[36:42], axis=0),  # Right eye
            np.mean(self.reference[42:48], axis=0),  # Left eye
            np.mean(self.reference[31:36], axis=0),  # Nose tip
            np.mean(self.reference[48:68], axis=0),  # Mouth center
        ])
        stable_ref[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
        stable_ref[:, 1] -= (reference_size[1] - target_size[1]) / 2.0
        return stable_ref

    def affine_transform(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        target_size: Tuple[int, int] = (256, 256),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Aligns frame with canonical reference landmarks using similarity transform."""
        stable_ref = self.get_stable_reference(target_size=target_size)
        transform = cv2.estimateAffinePartial2D(
            landmarks[:4],
            stable_ref,
            method=cv2.LMEDS,
        )[0]

        if transform is None:
            # Fallback to standard affine if partial 2D fails
            transform = cv2.getAffineTransform(
                landmarks[:3].astype(np.float32),
                stable_ref[:3].astype(np.float32),
            )

        transformed_frame = cv2.warpAffine(
            frame,
            transform,
            dsize=target_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        transformed_landmarks = (
            np.matmul(landmarks, transform[:, :2].T) + transform[:, 2].T
        )
        return transformed_frame, transformed_landmarks

    def cut_patch(
        self,
        img: np.ndarray,
        center: np.ndarray,
        half_height: int,
        half_width: int,
    ) -> np.ndarray:
        """Cuts a patch of size (2*half_height, 2*half_width) centered at center with safe padding."""
        cx, cy = int(round(center[0])), int(round(center[1]))
        h, w = img.shape[:2]

        y_min = cy - half_height
        y_max = cy + half_height
        x_min = cx - half_width
        x_max = cx + half_width

        # Calculate padding if box extends beyond image boundaries
        pad_top = max(0, -y_min)
        pad_bottom = max(0, y_max - h)
        pad_left = max(0, -x_min)
        pad_right = max(0, x_max - w)

        src_y_min = max(0, y_min)
        src_y_max = min(h, y_max)
        src_x_min = max(0, x_min)
        src_x_max = min(w, x_max)

        crop = img[src_y_min:src_y_max, src_x_min:src_x_max]

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            if img.ndim == 3:
                crop = np.pad(
                    crop,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode="edge",
                )
            else:
                crop = np.pad(
                    crop,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode="edge",
                )

        target_h, target_w = 2 * half_height, 2 * half_width
        if crop.shape[0] != target_h or crop.shape[1] != target_w:
            crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        return crop

    def __call__(
        self,
        video_frames: Union[np.ndarray, List[np.ndarray]],
        landmarks: List[np.ndarray],
    ) -> np.ndarray:
        """Processes video frames and returns aligned ROI sequence of shape (T, H, W, C)."""
        num_frames = len(video_frames)
        assert len(landmarks) == num_frames, "Number of frames and landmark sets must match"

        sequence = []
        for i, frame in enumerate(video_frames):
            # Temporal landmark smoothing over window_margin
            margin = min(self.window_margin // 2, i, num_frames - 1 - i)
            smoothed_lm = np.mean(
                [landmarks[idx] for idx in range(i - margin, i + margin + 1)],
                axis=0,
            )
            # Retain current frame relative position
            smoothed_lm += landmarks[i].mean(axis=0) - smoothed_lm.mean(axis=0)

            # Align frame
            transformed_frame, transformed_lm = self.affine_transform(frame, smoothed_lm)

            if self.representation == "full_face":
                patch = cv2.resize(transformed_frame, (self.crop_width, self.crop_height))
            elif self.representation == "mouth_jaw":
                # Slightly larger patch covering lower half of the face
                mouth_pt = transformed_lm[3]
                patch = self.cut_patch(
                    transformed_frame,
                    mouth_pt,
                    half_height=int(self.crop_height * 0.6),
                    half_width=self.crop_width // 2,
                )
            else:
                # Standard mouth-only patch (default 96x96 centered around mouth point [3])
                mouth_pt = transformed_lm[3]
                patch = self.cut_patch(
                    transformed_frame,
                    mouth_pt,
                    half_height=self.crop_height // 2,
                    half_width=self.crop_width // 2,
                )

            sequence.append(patch)

        return np.array(sequence, dtype=np.uint8)
