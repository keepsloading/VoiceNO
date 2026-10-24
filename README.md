# VoiceNO

> **You have the right to remain silent.**

**Silent input for your AI.** VoiceNO explores turning silently mouthed webcam input into text you can review and use in an AI tool.

VoiceNO is a camera-only silent-input research prototype. Recognition quality on genuinely silent speech is still under investigation.

## Start here

The current milestone is a measured silent-speech baseline. Your laptop captures video; your own interactive Colab GPU session performs heavy inference. Free GPU access is variable. No paid API or public inference service is required for this workflow.

1. Read [the baseline quickstart](docs/BASELINE_QUICKSTART.md).
2. Prepare the current source: `python experiments/package_eval_source.py`.
3. Upload [VoiceNO_Baseline.ipynb](experiments/VoiceNO_Baseline.ipynb) to Colab and select a GPU runtime.
4. Upload `voiceno_eval_source.zip`, then supply existing recordings or record new silent prompts in the notebook.
5. Download the JSON report containing actual predictions, failures, settings, and model identities.

Uploading recordings or recording through Colab sends video to that remote runtime. The prototype requests no microphone. Tight mouth cropping is data minimization; it does not establish anonymity, confidential computing, or guaranteed erasure.

## Local commands

From this directory, using an environment with `requirements.txt` installed:

```bash
python -m voiceno --help
python -m voiceno export-calib --user default
python experiments/package_eval_source.py
```

The earlier `python -m app.main ...` entry point is also supported. Configuration lives in `voiceno/config.py` as `VoiceNoConfig`; all application imports now use `voiceno`.

The experimental Gradio UI starts with `python -m voiceno ui`. Its local transcription/training features load the full model and may exceed this laptop's capacity. Use the Colab baseline as the primary workflow.

To create a separate dependency environment when needed:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

An existing activated Python environment with these dependencies can also be reused.

## What exists

- MediaPipe landmarks, aligned mouth crops, and Auto-AVSR inference.
- Separate personalization adapters and profile loading checks.
- Baseline/comparative evaluation with real predictions, failed-recording accounting, source/model/dataset identities, and saved candidates.
- A notebook-native Colab workflow and a portable source packager.
- Experimental context rescoring, clarification, desktop capture, and cloud modules.

Recognition quality and personalization benefits still need remote GPU evaluation on verified silent recordings. Current context scoring uses heuristics; confidence scores are uncalibrated. Some older UI features remain prototypes, including simulated agent execution. See [the MVP plan](docs/MVP_PLAN.md) for the ordered rebuild and [the inherited reference](docs/LEGACY_REFERENCE.md) for historical implementation details.

## Project layout

```text
voiceno/        Recognition, preprocessing, personalization, evaluation, branding
app/           CLI, experimental Gradio UI, capture utilities
experiments/   Colab notebooks, benchmark runners, source packager
tests/         Unit tests and a separate heavy integration test
docs/          Baseline instructions and MVP roadmap
```

## Verify without running the full model

```bash
python -m pytest tests/ --ignore=tests/test_end_to_end_personalization.py -q
```

The excluded integration test loads/trains the full model. Unit tests use tiny synthetic models to check software behavior; passing them is not evidence of speech recognition accuracy.

## Code and model rights

Project integration code retains [Apache 2.0](LICENSE) licensing and upstream notices. Auto-AVSR, bundled third-party code, tokenizer/preprocessing assets, and pretrained weights retain their respective provenance and terms. Commercial rights to the exact pretrained checkpoint require separate verification.

The project repository is [keepsloading/VoiceNO](https://github.com/keepsloading/VoiceNO). Product branding, code, and notebooks all use VoiceNO.
