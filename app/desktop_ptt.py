"""Desktop Push-to-Talk (PTT) Companion Client for VoiceNO.

Runs locally in the user's background / menubar.
Workflow:
1. User presses and holds Push-To-Talk hotkey (e.g. Fn / CapsLock / Cmd+Shift+Space).
2. Local webcam captures silent articulation at 25 FPS.
3. Local EdgeMouthExtractor crops and normalizes 88x88 mouth patch (<300 KB payload).
   Full face, eyes, hair, and room are immediately discarded locally.
4. Serialized payload is sent to Confidential Cloud Endpoint or local model.
5. Emits decoded prompt text directly to cursor in active app (Cursor, Claude, ChatGPT, Terminal).
"""

import argparse
import json
import time
import urllib.request
from typing import Any, Dict, Optional
import cv2
import numpy as np

from voiceno.client.mouth_extractor import EdgeMouthExtractor
from voiceno.domains import get_active_domain


class DesktopPTTClient:
    """Manages local webcam capture, edge mouth extraction, and cloud inference transmission."""

    def __init__(self, endpoint_url: Optional[str] = None):
        self.endpoint_url = endpoint_url or "http://localhost:8000"
        self.extractor = EdgeMouthExtractor()

    def record_ptt_clip(self, duration_sec: float = 2.5, camera_id: int = 0) -> List[np.ndarray]:
        """Captures camera frames for the duration of the push-to-talk press."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}.")

        frames = []
        fps = 25.0
        max_frames = int(duration_sec * fps)

        print(f"🎙️ [PTT ACTIVE] Recording silent mouth movement for {duration_sec}s...")
        t0 = time.time()
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            time.sleep(1.0 / fps)

        cap.release()
        print(f"⏹️ [PTT RELEASED] Captured {len(frames)} frames in {time.time() - t0:.2f}s.")
        return frames

    def process_and_send(self, frames: List[np.ndarray], domain: Optional[str] = None) -> Dict[str, Any]:
        """Strips face locally and transmits 300 KB mouth payload to inference server."""
        print("🔒 Stripping face locally with MediaPipe (Eyes & room discarded)...")
        payload_bytes, telemetry = self.extractor.extract_anonymized_payload(frames)
        print(f"📦 Anonymized payload ready: {telemetry['payload_size_kb']} KB.")

        # Post to endpoint
        target_domain = domain or get_active_domain().name
        url = f"{self.endpoint_url}/transcribe?domain={target_domain}"

        req = urllib.request.Request(
            url,
            data=payload_bytes,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )

        t_send = time.time()
        with urllib.request.urlopen(req) as resp:
            resp_body = resp.read().decode("utf-8")
            result = json.loads(resp_body)
        latency_total = (time.time() - t_send) * 1000

        result["client_telemetry"] = telemetry
        result["network_latency_ms"] = round(latency_total, 2)
        return result


def main():
    parser = argparse.ArgumentParser(description="VoiceNO Desktop Push-To-Talk Client")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000", help="Cloud / Local inference server URL")
    parser.add_argument("--domain", type=str, default="ai_prompts", choices=["ai_prompts", "coding", "email", "blogging"], help="Active domain")
    parser.add_argument("--duration", type=float, default=2.5, help="Recording duration in seconds")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device ID")
    args = parser.parse_args()

    client = DesktopPTTClient(endpoint_url=args.endpoint)
    frames = client.record_ptt_clip(duration_sec=args.duration, camera_id=args.camera)
    result = client.process_and_send(frames, domain=args.domain)

    print("\n" + "=" * 50)
    print(f"DECODED SILENT PROMPT: {result.get('text', '')}")
    print(f"DOMAIN:                {result.get('domain', '')}")
    print(f"CONFIDENCE:            {result.get('confidence', '')}")
    print(f"SERVER LATENCY:        {result.get('latency_ms', 0)} ms")
    print("=" * 50)


if __name__ == "__main__":
    main()
