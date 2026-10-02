"""Webcam capture module for VoiceNo! V0.

Captures camera video frames without microphone audio.
PRIVACY-FIRST: Absolutely zero microphone or audio device recording.
Supports 720p+ input, target ~25 FPS, with graceful fallback to video file for offline testing.
"""

import threading
import time
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union
import cv2
import numpy as np


class WebcamCapture:
    """Thread-safe camera-only video stream."""

    def __init__(
        self,
        device_index: int = 0,
        target_width: int = 1280,
        target_height: int = 720,
        target_fps: float = 25.0,
        video_source_fallback: Optional[Union[str, Path]] = None,
    ):
        self.device_index = device_index
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps
        self.video_source_fallback = video_source_fallback

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running: bool = False
        self._lock = threading.Lock()
        self._current_frame: Optional[np.ndarray] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """Starts video capture stream."""
        with self._lock:
            if self.is_running:
                return True

            # Attempt to open physical webcam device
            self.cap = cv2.VideoCapture(self.device_index)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            elif self.video_source_fallback and Path(self.video_source_fallback).is_file():
                # Fallback to local video file for testing
                self.cap = cv2.VideoCapture(str(self.video_source_fallback))

            if not self.cap or not self.cap.isOpened():
                return False

            self.is_running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            return True

    def _capture_loop(self):
        frame_interval = 1.0 / self.target_fps
        while self.is_running and self.cap and self.cap.isOpened():
            t0 = time.time()
            ret, frame = self.cap.read()
            if not ret:
                # If looped video file, rewind
                if self.video_source_fallback:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._current_frame = rgb_frame

            elapsed = time.time() - t0
            sleep_time = max(0.001, frame_interval - elapsed)
            time.sleep(sleep_time)

    def read_frame(self) -> Optional[np.ndarray]:
        """Reads latest frame as RGB numpy array."""
        with self._lock:
            return self._current_frame.copy() if self._current_frame is not None else None

    def record_utterance(self, duration_seconds: float = 3.0) -> List[np.ndarray]:
        """Records an utterance of specified duration at target FPS."""
        frames = []
        frame_interval = 1.0 / self.target_fps
        total_frames = int(duration_seconds * self.target_fps)

        for _ in range(total_frames):
            t0 = time.time()
            frame = self.read_frame()
            if frame is not None:
                frames.append(frame)
            elapsed = time.time() - t0
            time.sleep(max(0.001, frame_interval - elapsed))

        return frames

    def stop(self):
        """Stops video capture and releases camera hardware."""
        with self._lock:
            self.is_running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self._current_frame = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
