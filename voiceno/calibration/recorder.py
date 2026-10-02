"""Calibration recorder for VoiceNo! V0.

Captures camera-only video utterances with displayed prompts.
PRIVACY-FIRST: Strictly prohibits microphone recording or audio device access.
Stores data in structured calibration vs evaluation partitions.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from voiceno.config import REPO_ROOT


class CalibrationRecorder:
    """Manages recording and persistence of calibration and evaluation sessions."""

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        self.base_dir = Path(base_dir or (REPO_ROOT / "calibration_data"))
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_user_dir(self, user_id: str, partition: str = "calibration") -> Path:
        """Returns partition directory for a given user ('calibration' or 'evaluation')."""
        assert partition in ("calibration", "evaluation"), "Partition must be 'calibration' or 'evaluation'"
        user_dir = self.base_dir / user_id / partition
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    def save_utterance(
        self,
        user_id: str,
        frames: List[np.ndarray],
        prompt_text: str,
        partition: str = "calibration",
        fps: float = 25.0,
    ) -> Dict:
        """Saves a single recorded video utterance with ground-truth prompt text.

        Camera-only: no audio is recorded or stored.
        """
        user_dir = self.get_user_dir(user_id, partition)
        timestamp = int(time.time() * 1000)
        utterance_id = f"utt_{timestamp}"

        video_path = user_dir / f"{utterance_id}.mp4"
        meta_path = user_dir / f"{utterance_id}.json"

        if len(frames) > 0:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
            for f in frames:
                # Expect RGB input, write BGR for OpenCV
                bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR) if f.shape[-1] == 3 else f
                writer.write(bgr)
            writer.release()

        metadata = {
            "utterance_id": utterance_id,
            "user_id": user_id,
            "prompt": prompt_text.upper().strip(),
            "partition": partition,
            "num_frames": len(frames),
            "fps": fps,
            "duration_seconds": round(len(frames) / fps, 2),
            "timestamp": timestamp,
            "video_path": str(video_path),
        }

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def load_dataset(self, user_id: str, partition: str = "calibration") -> List[Dict]:
        """Loads all recorded utterance metadata for a user partition."""
        user_dir = self.get_user_dir(user_id, partition)
        utterances = []
        for meta_file in sorted(user_dir.glob("*.json")):
            with open(meta_file) as f:
                utterances.append(json.load(f))
        return utterances

    def export_calibration_package(
        self,
        user_id: str,
        output_zip_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Packages a user's calibration and evaluation utterances into a portable zip.

        This zip can be uploaded to Google Colab or a GPU server for fast cloud LoRA training.
        """
        import zipfile
        user_root = self.base_dir / user_id
        if not user_root.is_dir():
            raise FileNotFoundError(f"No calibration data found for user '{user_id}' at: {user_root}")

        if output_zip_path is None:
            output_zip_path = self.base_dir / f"{user_id}_calibration_package.zip"
        else:
            output_zip_path = Path(output_zip_path)

        with zipfile.ZipFile(str(output_zip_path), "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in user_root.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(user_root)
                    zipf.write(str(file_path), str(arcname))

        return output_zip_path

