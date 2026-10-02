"""Temporal sequence normalization for Auto-AVSR input.

Converts cropped mouth patches into normalized grayscale tensors (T, 1, 88, 88)
matching the exact Auto-AVSR training pipeline distribution.
"""

from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torchvision.transforms as T

from voiceno.config import PreprocessingConfig
from voiceno.preprocessing.face import FaceDetector
from voiceno.preprocessing.mouth_roi import MouthROIExtractor


class VideoNormalizer:
    """Normalizes video frames to grayscale tensors matching Auto-AVSR specifications."""

    def __init__(
        self,
        target_size: int = 88,
        mean: float = 0.421,
        std: float = 0.165,
    ):
        self.target_size = target_size
        self.mean = mean
        self.std = std
        self.center_crop = T.CenterCrop(target_size)

    def __call__(self, sequence: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Transforms (T, H, W, C) or (T, H, W) numpy array to (T, 1, target_size, target_size) float32 tensor.

        Pipeline:
        1. Convert to RGB / Grayscale
        2. Scale [0, 255] -> [0.0, 1.0]
        3. CenterCrop to target_size (e.g. 88x88)
        4. Normalize by (mean, std)
        """
        if isinstance(sequence, np.ndarray):
            # If shape is (T, H, W, 3), convert to (T, 3, H, W)
            if sequence.ndim == 4:
                # Convert RGB to Grayscale
                gray_seq = np.array([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if f.shape[-1] == 3 else f for f in sequence])
            elif sequence.ndim == 3:
                gray_seq = sequence
            else:
                raise ValueError(f"Unexpected sequence shape: {sequence.shape}")

            tensor = torch.from_numpy(gray_seq).float()  # (T, H, W)
            tensor = tensor.unsqueeze(1)  # (T, 1, H, W)
        else:
            tensor = sequence.float()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(1)

        # Scale to [0, 1]
        if tensor.max() > 1.0:
            tensor = tensor / 255.0

        # Center crop from 96x96 to 88x88
        tensor = self.center_crop(tensor)

        # Normalize with dataset statistics
        tensor = (tensor - self.mean) / self.std

        return tensor


class VideoPreprocessor:
    """End-to-end preprocessing pipeline from raw webcam frames to model-ready tensor."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.face_detector = FaceDetector()
        self.roi_extractor = MouthROIExtractor(
            mean_face_path=self.config.mean_face_path,
            crop_width=self.config.crop_width,
            crop_height=self.config.crop_height,
            window_margin=self.config.window_margin,
            representation=self.config.representation,
        )
        self.normalizer = VideoNormalizer(
            target_size=self.config.model_input_size,
            mean=self.config.norm_mean,
            std=self.config.norm_std,
        )

    def process(self, video_frames: Union[np.ndarray, List[np.ndarray]]) -> Tuple[torch.Tensor, np.ndarray]:
        """Runs complete preprocessing pipeline.

        Returns:
            tensor: torch.Tensor of shape (T, 1, 88, 88)
            crops: np.ndarray of shape (T, 96, 96, 3) for preview / logging
        """
        raw_landmarks = self.face_detector(video_frames)
        interpolated_landmarks = self.face_detector.interpolate_landmarks(raw_landmarks)

        if interpolated_landmarks is None:
            raise RuntimeError("No face detected in video sequence")

        mouth_crops = self.roi_extractor(video_frames, interpolated_landmarks)
        normalized_tensor = self.normalizer(mouth_crops)

        return normalized_tensor, mouth_crops
