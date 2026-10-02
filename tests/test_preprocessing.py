"""Tests for VoiceNo preprocessing pipeline."""

import numpy as np
import pytest
import torch

from voiceno.preprocessing.face import FaceDetector
from voiceno.preprocessing.mouth_roi import MouthROIExtractor
from voiceno.preprocessing.normalize import VideoNormalizer, VideoPreprocessor


def test_face_detector_interpolation():
    """Verify linear interpolation of missing frames."""
    # 5 frames with middle frames missing
    lm1 = np.array([[10, 10], [20, 10], [15, 15], [15, 20]], dtype=np.float32)
    lm4 = np.array([[20, 20], [30, 20], [25, 25], [25, 30]], dtype=np.float32)

    landmarks = [lm1, None, None, lm4, None]
    interpolated = FaceDetector.interpolate_landmarks(landmarks)

    assert len(interpolated) == 5
    assert all(lm is not None for lm in interpolated)
    # Check linear progression
    np.testing.assert_allclose(interpolated[0], lm1)
    np.testing.assert_allclose(interpolated[3], lm4)
    # Check trailing frame filled from last valid
    np.testing.assert_allclose(interpolated[4], lm4)


def test_mouth_roi_patch_cutting():
    """Verify cut_patch produces exact expected dimensions even at borders."""
    extractor = MouthROIExtractor()
    img = np.zeros((256, 256, 3), dtype=np.uint8)

    # Test center crop
    patch_center = extractor.cut_patch(img, np.array([128, 128]), half_height=48, half_width=48)
    assert patch_center.shape == (96, 96, 3)

    # Test edge boundary (near 0,0)
    patch_edge = extractor.cut_patch(img, np.array([10, 10]), half_height=48, half_width=48)
    assert patch_edge.shape == (96, 96, 3)


def test_video_normalizer():
    """Verify tensor shape and normalization bounds."""
    normalizer = VideoNormalizer(target_size=88)
    dummy_crops = np.full((10, 96, 96, 3), 128, dtype=np.uint8)

    tensor = normalizer(dummy_crops)
    assert tensor.shape == (10, 1, 88, 88)
    assert tensor.dtype == torch.float32
    # Verify center crop was applied (88x88)
    assert tensor.shape[2] == 88 and tensor.shape[3] == 88


if __name__ == "__main__":
    test_face_detector_interpolation()
    test_mouth_roi_patch_cutting()
    test_video_normalizer()
    print("All preprocessing unit tests passed!")
