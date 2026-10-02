# VoiceNO!

> **You have the right to remain silent.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Status: V0 Research Prototype](https://img.shields.io/badge/Status-V0_Offline_Prototype-orange.svg)](#current-status)

**VoiceNO** is a camera-only silent speech recognition prototype with personalized visual speech adaptation.

---

## 1. What is VoiceNO?

VoiceNO performs silent visual speech recognition (lip-reading) entirely from camera video without a microphone. It pairs a pretrained visual speech recognition (VSR) foundation model with parameter-efficient Low-Rank Adaptation (LoRA) to study one core research question:

> **Can a pretrained visual speech recognition model become better at recognizing one person's speech after a few minutes of personalized calibration?**

---

## 2. Why VoiceNO?

Conventional automated speech recognition (ASR) relies exclusively on acoustic signals from a microphone. Acoustic speech recognition fails in high-noise environments, poses privacy risks in public spaces, and does not serve individuals with vocal cord conditions or laryngectomy. 

VoiceNO explores whether camera-only visual speech signals can transcribe silent articulation accurately, and demonstrates how personalized adaptation bridges the gap between generic foundation models and individual facial articulatory dynamics.

---

## 3. Current Status

```text
Status: V0 — Offline Personalized VSR Prototype
```

> [!IMPORTANT]
> **What V0 is:**
> - A functional offline utterance-based research prototype.
> - An end-to-end pipeline: Camera Video &rarr; MediaPipe Face Landmarks &rarr; Normalized 96&times;96 Mouth ROI &rarr; Auto-AVSR Backbone &rarr; LoRA Personalization &rarr; Text.
> - A modular evaluation harness comparing Generic vs. Personalized performance on held-out utterances using reproducible WER and CER metrics.
>
> **What V0 is NOT:**
> - Not real-time streaming (processes recorded visual utterances offline).
> - Not production-ready or commercially cleared.
> - Not claimed to be universal or robust across all camera angles and lighting conditions.

---

## 4. Architecture Pipeline

```text
Webcam Video (720p+, ~25 FPS, RGB)
        ↓
Face Detection & Landmarks (MediaPipe)
        ↓
Mouth ROI Extraction & Affine Normalization
        ↓
96×96 Grayscale Normalization (CenterCrop 88×88, (x-μ)/σ)
        ↓
Pretrained VSR Backbone (Auto-AVSR Conformer)
        ↓
Personalization Layer (Frozen Base + LoRA on Wq, Wk, Wv; r=8)
        ↓
Transcribed Text (CTC / Beam Search)
```

---

## 5. Technology Stack

- **Language & Runtime:** Python 3.12, PyTorch (CPU & GPU compatible)
- **Computer Vision:** OpenCV, MediaPipe FaceLandmarker, torchvision
- **Pretrained VSR Backbone:** Auto-AVSR (`vsr_trlrs2lrs3vox2avsp_base.pth`, 3,291-hour checkpoint)
- **Personalization:** Parameter-Efficient LoRA ($r=8$, $\alpha=16$) on attention projections ($W_q, W_k, W_v$)
- **Evaluation Metrics:** Word Error Rate (WER) & Character Error Rate (CER) via JiWER
- **User Interface:** Local Gradio Web Interface

---

## 6. Privacy & Safety Design

VoiceNO is designed with a strict **privacy-first** architecture:
- **No Microphone Access:** Audio hardware is never opened, queried, or recorded.
- **Local Inference:** All frame processing and model inference execute strictly locally on the user's machine. No visual data is transmitted over networks.
- **Explicit Storage:** Webcam video is never persisted silently. Utterances are only stored when the user explicitly initiates a calibration or evaluation recording.
- **Minimal Footprint:** Only the tiny personalization delta weights (`lora.pt`, ~1.7 MB) are saved to user profiles, preserving the immutable base checkpoint.

---

## 7. Installation & Quickstart

### Prerequisites

- Linux (tested on Fedora Linux, Ubuntu)
- Python 3.12
- Webcam or local video file

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/keepsloading/VoiceNO.git
cd VoiceNO

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Pretrained Checkpoint Setup

The Auto-AVSR 3,291-hour visual checkpoint (`vsr_trlrs2lrs3vox2avsp_base.pth`) is placed in `checkpoints/`:

```bash
mkdir -p checkpoints
# Download via gdown if not already present
gdown 1r1kx7l9sWnDOCnaFHIGvOtzuhFyFA88_ -O checkpoints/vsr_trlrs2lrs3vox2avsp_base.pth
```

Verify checkpoint integrity:
- Expected MD5: `49f770f2c0d8b8d769347ee47ed1648f`

---

## 8. Usage

### A. Launch Local Web Interface

```bash
python -m app.main ui --port 7860
```
Open `http://localhost:7860` in your web browser. The interface allows you to:
1. Record or upload silent utterances for instant transcription.
2. Step through calibration sentences to personalize the model.
3. Run comparative held-out benchmark evaluations.

### B. Command-Line Interface (CLI)

#### 1. Transcribe a Video (Generic Baseline)
```bash
python -m app.main transcribe --video tests/benchmark_sample.mp4 --mode generic
```

#### 2. Transcribe with Personalized Profile
```bash
python -m app.main transcribe --video tests/benchmark_sample.mp4 --mode personalized --user default
```

#### 3. Run User Calibration (Personalization Training)
```bash
# Option A: Fast CPU Adapter (Instant, <50MB RAM - Recommended for laptops)
python -m app.main calibrate --user user_001 --strategy fast_cpu --epochs 10

# Option B: Full Conformer LoRA (12 Attention layers - For Cloud / GPU machines)
python -m app.main calibrate --user user_001 --strategy conformer_lora --epochs 10
```

#### 4. Export Calibration Package for Free Google Colab GPU
```bash
python -m app.main export-calib --user user_001
# Open experiments/VoiceNO_Colab_Calibration.ipynb in Google Colab, upload the zip,
# train on a free T4 GPU in 20 seconds, and download the lora.pt back to your laptop!
```

#### 5. Run Comparative Evaluation (Held-Out Data)
```bash
python -m app.main evaluate --user user_001 --calib-min 5.0
```

#### 6. Run Calibration Duration Scaling Experiment
```bash
python experiments/run_duration_experiment.py --user user_001
```

---

## 9. Personalization & Calibration Workflow (Hardware-Aware)

Standard transformer backpropagation through 12 Conformer layers on video requires 3-4 GB of activation memory, which causes severe memory thrashing and thermal throttling on ultraportable laptop CPUs (e.g. 15W quad-core CPUs with 6-8 GB RAM).

VoiceNO solves this with two complementary training architectures:

1. **⚡ Fast CPU Adapter (Instant Local Calibration):**
   - Extracts Conformer embeddings `(T, 768)` in `torch.no_grad()` (< 0.8s per utterance).
   - Trains a low-rank adapter on the CTC projection head (`model.ctc.ctc_lo`) using CTC loss.
   - **RAM footprint:** < 50 MB (zero swap thrashing).
   - **Training speed:** ~0.3s to 2s on laptop CPU!
   - Ideal for immediate, private, fully offline calibration on any laptop.

2. **☁️ Free Cloud GPU Offload (Google Colab 1-Click):**
   - Click **"Export Calibration Package (.zip)"** in the UI or run `python -m app.main export-calib`.
   - Open [`experiments/VoiceNO_Colab_Calibration.ipynb`](experiments/VoiceNO_Colab_Calibration.ipynb) in Google Colab.
   - Fine-tunes all 12 multi-head attention layers ($W_q, W_k, W_v$) on a free Nvidia T4 GPU in ~20 seconds.
   - Automatically downloads the tiny `lora.pt` (1.7 MB) back to your laptop `user_profiles/{user_id}/` folder.
   - Local VoiceNO inference remains 100% private and offline!


Personalization enables user-specific visual adaptation without modifying the foundational weights:

1. **Prompt Display:** VoiceNO presents phonetically rich sentences for silent articulation.
2. **Calibration Partition:** Utterances are recorded into `calibration_data/{user_id}/calibration/`.
3. **Held-Out Evaluation:** A strictly disjoint set of sentences is recorded into `calibration_data/{user_id}/evaluation/`.
4. **LoRA Fine-Tuning:** The Auto-AVSR base weights remain frozen. Only low-rank matrices $A$ and $B$ attached to attention projections ($W_q, W_k, W_v$) receive gradients.
5. **Profile Storage:** Personalization weights are saved as `user_profiles/{user_id}/lora.pt`.
6. **Comparative Verification:** Both Generic and Personalized models are evaluated on the exact same held-out evaluation utterances, outputting machine-readable JSON metrics:

```json
{
  "user": "user_001",
  "calibration_minutes": 5.0,
  "wer": 0.2857,
  "cer": 0.1245,
  "inference_seconds": 2.74,
  "generic_baseline": {
    "wer": 0.3810,
    "cer": 0.1782,
    "inference_seconds": 2.72
  },
  "personalized": {
    "wer": 0.2857,
    "cer": 0.1245,
    "inference_seconds": 2.74
  },
  "absolute_improvements": {
    "wer_reduction": 0.0953,
    "cer_reduction": 0.0537
  }
}
```

---

## 10. Repository Structure

```text
voiceno/
├── app/
│   ├── main.py                     # CLI entrypoint and subcommand dispatcher
│   ├── webcam.py                   # Camera-only video capture (audio-free)
│   └── ui.py                       # Gradio local web interface
│
├── voiceno/
│   ├── preprocessing/
│   │   ├── face.py                 # MediaPipe FaceLandmarker tracking & interpolation
│   │   ├── mouth_roi.py            # Affine alignment & 96x96 mouth ROI extraction
│   │   ├── normalize.py            # CenterCrop 88x88 & distribution normalization
│   │   └── 20words_mean_face.npy   # Canonical 68-point reference face landmarks
│   │
│   ├── models/
│   │   ├── auto_avsr.py            # Auto-AVSR Conformer wrapper & CTC/Beam decoding
│   │   ├── personalization.py      # LoRA engine & PromptAdapter
│   │   ├── espnet/                 # E2E Conformer model definition
│   │   └── spm/                    # Unigram SentencePiece tokenizer & vocab
│   │
│   ├── calibration/
│   │   ├── prompts.py              # Disjoint calibration & evaluation prompt corpora
│   │   ├── recorder.py             # Audio-free video utterance partition manager
│   │   └── trainer.py              # Frozen-base LoRA fine-tuning loop
│   │
│   ├── evaluation/
│   │   ├── metrics.py              # Normalized WER & CER calculation
│   │   └── evaluate.py             # Generic vs Personalized comparative benchmark
│   │
│   ├── context/
│   │   └── rescorer.py             # Contextual disambiguation & linguistic rescorer
│   │
│   ├── clarification/
│   │   └── manager.py              # Interactive clarification manager for ambiguous speech
│   │
│   └── config.py                   # Centralized dataclass configurations
│
├── checkpoints/                    # Foundation model weights
├── calibration_data/               # Recorded calibration & evaluation sessions
├── user_profiles/                  # Saved user-specific LoRA deltas
├── experiments/                    # Machine-readable experiment results & sweeps
├── tests/                          # Automated unit and integration test suite
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 11. Testing

Run the automated test suite:

```bash
# Run all unit and integration tests
pytest tests/ -v
```

Test coverage includes:
- `tests/test_preprocessing.py`: Landmark interpolation, boundary-safe patch cutting, normalizer dimensions.
- `tests/test_personalization.py`: LoRA zero-delta initialization, base parameter freezing, profile save & load.
- `tests/test_metrics.py`: Normalized WER & CER computation, edge cases.
- `tests/test_end_to_end_personalization.py`: Complete lifecycle from recording to LoRA fine-tuning and held-out comparative evaluation.

---

## 12. Licensing & Disclosures

### Code License
The VoiceNO software and integration code are licensed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for details.

### Research Model Checkpoint Licensing Disclosure
The pretrained Visual Speech Recognition checkpoint `vsr_trlrs2lrs3vox2avsp_base.pth` is derived from the **Auto-AVSR** research project (Imperial College London / Pingchuan Ma et al.).

- **Code:** Apache 2.0 licensed.
- **Training Data & Weights:** This checkpoint was trained on research datasets including LRS2, LRS3-TED, VoxCeleb2, and AVSpeech. These datasets are subject to individual non-commercial and research-only academic license terms.
- **Disclaimer:** VoiceNO is an **academic research prototype**. The use of this checkpoint is strictly for research exploration and benchmarking. No commercial rights, endorsements, or clearances are claimed or implied.
