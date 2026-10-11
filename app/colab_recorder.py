"""In-browser WebRTC webcam recorder and inference pipeline runner for Google Colab."""

import base64
import os
from pathlib import Path
import subprocess
from typing import Any, Dict


def record_webcam(output_file: str = "webcam_test.mp4") -> str:
    """Renders a Push-to-Talk WebRTC webcam interface in Colab and records a 25 FPS MP4."""
    from google.colab import output

    js_code = """
    async function recordVideo() {
        const div = document.createElement("div");
        div.style.padding = "14px";
        div.style.border = "1px solid #444";
        div.style.borderRadius = "8px";
        div.style.background = "#181818";
        div.style.color = "#fff";
        div.style.width = "fit-content";
        
        const video = document.createElement("video");
        video.style.display = "block";
        video.style.borderRadius = "6px";
        video.width = 480;
        video.height = 360;
        video.autoplay = true;
        video.muted = true;
        video.playsInline = true;
        
        const btnRecord = document.createElement("button");
        btnRecord.innerText = "Start Recording";
        btnRecord.style.margin = "10px 8px 6px 0";
        btnRecord.style.padding = "8px 16px";
        btnRecord.style.fontWeight = "bold";
        btnRecord.style.cursor = "pointer";
        btnRecord.style.backgroundColor = "#d32f2f";
        btnRecord.style.color = "#fff";
        btnRecord.style.border = "none";
        btnRecord.style.borderRadius = "4px";

        const btnStop = document.createElement("button");
        btnStop.innerText = "Stop and Transcribe";
        btnStop.disabled = true;
        btnStop.style.margin = "10px 8px 6px 0";
        btnStop.style.padding = "8px 16px";
        btnStop.style.fontWeight = "bold";
        btnStop.style.cursor = "pointer";
        btnStop.style.backgroundColor = "#555";
        btnStop.style.color = "#fff";
        btnStop.style.border = "none";
        btnStop.style.borderRadius = "4px";

        const status = document.createElement("span");
        status.innerText = "Initializing camera...";
        status.style.fontFamily = "monospace";

        div.appendChild(video);
        div.appendChild(btnRecord);
        div.appendChild(btnStop);
        div.appendChild(status);
        document.querySelector("#output-area").appendChild(div);

        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
        video.srcObject = stream;
        status.innerText = "Ready. Click Start Recording and mouth words clearly.";

        const recorder = new MediaRecorder(stream, { mimeType: "video/webm" });
        const chunks = [];
        recorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };

        return new Promise((resolve) => {
            btnRecord.onclick = () => {
                chunks.length = 0;
                recorder.start();
                btnRecord.disabled = true;
                btnRecord.style.backgroundColor = "#555";
                btnStop.disabled = false;
                btnStop.style.backgroundColor = "#2e7d32";
                status.innerText = "Recording... mouth your words now!";
            };

            btnStop.onclick = () => {
                recorder.stop();
                stream.getTracks().forEach(t => t.stop());
                btnStop.disabled = true;
                btnStop.style.backgroundColor = "#555";
                status.innerText = "Resampling video to 25 FPS...";
            };

            recorder.onstop = () => {
                const blob = new Blob(chunks, { type: "video/webm" });
                const reader = new FileReader();
                reader.onloadend = () => resolve(reader.result);
                reader.readAsDataURL(blob);
            };
        });
    }
    recordVideo();
    """
    data_url = output.eval_js(js_code)
    header, encoded = data_url.split(",", 1)
    raw_bytes = base64.b64decode(encoded)

    temp_webm = "temp_webcam.webm"
    with open(temp_webm, "wb") as f:
        f.write(raw_bytes)

    # Standardize to 25.0 FPS MP4 for Auto-AVSR Conformer
    subprocess.run([
        "ffmpeg", "-y", "-i", temp_webm, "-r", "25",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", output_file
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return output_file


def get_model_and_rescorer():
    """Initializes and returns AutoAVSRModel and ContextRescorer, downloading checkpoint if needed."""
    import cv2
    import numpy as np
    import torch
    import torchvision.io

    def _read_video_cv2(filename, *args, **kwargs):
        cap = cv2.VideoCapture(filename)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise ValueError(f"Could not read frames from: {filename}")
        return torch.from_numpy(np.array(frames, dtype=np.uint8)), None, None

    torchvision.io.read_video = _read_video_cv2

    ckpt_path = Path("checkpoints/vsr_trlrs2lrs3vox2avsp_base.pth")
    if not ckpt_path.is_file():
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        print("Auto-AVSR base weights not found. Downloading (~1.0 GB)...")
        try:
            import gdown
            gdown.download(id="1r1kx7l9sWnDOCnaFHIGvOtzuhFyFA88_", output=str(ckpt_path), quiet=False)
        except Exception:
            subprocess.run(["gdown", "1r1kx7l9sWnDOCnaFHIGvOtzuhFyFA88_", "-O", str(ckpt_path)], check=True)

    from voiceno.config import ModelConfig
    from voiceno.models.auto_avsr import AutoAVSRModel
    from voiceno.context.rescorer import ContextRescorer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading VoiceNO pipeline on {device} (beam_size=40)...")
    config = ModelConfig(device=device, beam_size=40)
    model = AutoAVSRModel(config=config)
    model.load_checkpoint(str(ckpt_path))
    rescorer = ContextRescorer()
    print("VoiceNO Model and Rescorer ready!")
    return model, rescorer


def run_pipeline(
    model: Any = None,
    rescorer: Any = None,
    context: Any = None,
    output_file: str = "webcam_test.mp4",
    nbest: int = 15,
) -> Dict[str, Any]:
    """Records a clip from the browser webcam, runs Conformer beam search, and rescores hypotheses.

    If model or rescorer is not provided, automatically loads/initializes them.
    """
    if model is None or rescorer is None:
        loaded_model, loaded_rescorer = get_model_and_rescorer()
        model = model or loaded_model
        rescorer = rescorer or loaded_rescorer

    if context is None:
        from voiceno.context.rescorer import AgentContext
        context = AgentContext(
            domain_keywords=[
                "CONVERT", "IMAGE", "PDF", "SUMMARIZE", "DOCUMENT",
                "FILE", "CODE", "TESTS", "BUG", "OPEN", "EXPLAIN",
                "CALENDAR", "BROWSER", "SEARCH", "PLAYING", "BALL", "PARK", "MALL"
            ]
        )

    video_clip = record_webcam(output_file)

    print("Extracting mouth movements & running Conformer beam search (beam_size=40)...")
    details = model.transcribe(video_clip, use_beam_search=True, nbest=nbest)
    hypotheses = details.get("hypotheses", [])

    rescored = rescorer.rescore_hypotheses(hypotheses, context=context)

    top_vis = hypotheses[0]["text"] if hypotheses else ""
    top_rescored = rescored[0]["text"] if rescored else ""

    print("\n" + "=" * 55)
    print(f" TOP VISUAL HYPOTHESIS : {top_vis}")
    print(f" FINAL RESCORED TEXT   : {top_rescored}")
    print("=" * 55 + "\n")

    print("--- Candidate Beam Lattice ---")
    for i, h in enumerate(rescored):
        vis_s = h.get("visual_score", 0.0)
        lm_s = h.get("linguistic_score", 0.0)
        tot_s = h.get("rescored_score", 0.0)
        prob_s = h.get("rescored_prob", 0.0) * 100.0
        cand_text = h.get("text", "")
        print(f"[{i+1}] {cand_text} (Visual: {vis_s:.2f}, LM: {lm_s:.2f}, Total: {tot_s:.2f}, Prob: {prob_s:.1f}%)")

    return {
        "top_visual": top_vis,
        "final_rescored": top_rescored,
        "beam": rescored,
        "video_path": video_clip,
    }

