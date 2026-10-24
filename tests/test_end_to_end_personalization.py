"""End-to-end integration test for calibration, LoRA training, and evaluation.

Validates Milestones 5 through 9:
- Calibration data collection
- LoRA adapter fine-tuning on frozen Auto-AVSR base
- Independent adapter loading
- Held-out comparative evaluation (WER/CER)
- Reproducible JSON artifact generation
"""

import json
from pathlib import Path
import cv2
import pytest
import torch

from voiceno.calibration.prompts import CALIBRATION_PROMPTS, EVALUATION_PROMPTS
from voiceno.calibration.recorder import CalibrationRecorder
from voiceno.calibration.trainer import PersonalizationTrainer
from voiceno.config import REPO_ROOT, VoiceNoConfig
from voiceno.evaluation.evaluate import Evaluator
from voiceno.models.auto_avsr import AutoAVSRModel
from voiceno.models.personalization import PersonalizationManager


def test_complete_personalization_lifecycle(tmp_path):
    # 1. Setup config with temporary storage
    config = VoiceNoConfig()
    test_calib_dir = tmp_path / "calibration_data"
    test_profiles_dir = tmp_path / "user_profiles"
    test_experiments_dir = tmp_path / "experiments"

    recorder = CalibrationRecorder(base_dir=test_calib_dir)

    # 2. Slice benchmark sample video into calibration and held-out evaluation utterances
    cap = cv2.VideoCapture(str(REPO_ROOT / "tests" / "benchmark_sample.mp4"))
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    assert len(all_frames) >= 100, "Benchmark video must have at least 100 frames"

    # Calibration utterances (frames 0..50 and 50..100)
    calib_meta_1 = recorder.save_utterance(
        user_id="test_subject_01",
        frames=all_frames[0:50],
        prompt_text=CALIBRATION_PROMPTS[0],
        partition="calibration",
    )
    calib_meta_2 = recorder.save_utterance(
        user_id="test_subject_01",
        frames=all_frames[50:100],
        prompt_text=CALIBRATION_PROMPTS[1],
        partition="calibration",
    )

    # Held-out evaluation utterances (frames 100..150 and 150..200) - NEVER seen during training
    eval_meta_1 = recorder.save_utterance(
        user_id="test_subject_01",
        frames=all_frames[100:150],
        prompt_text=EVALUATION_PROMPTS[0],
        partition="evaluation",
    )
    eval_meta_2 = recorder.save_utterance(
        user_id="test_subject_01",
        frames=all_frames[150:200],
        prompt_text=EVALUATION_PROMPTS[1],
        partition="evaluation",
    )

    # Milestone 5: Calibration data collected
    calib_dataset = recorder.load_dataset("test_subject_01", partition="calibration")
    eval_dataset = recorder.load_dataset("test_subject_01", partition="evaluation")
    assert len(calib_dataset) == 2
    assert len(eval_dataset) == 2

    # 3. Load Auto-AVSR model
    model = AutoAVSRModel(config.model)
    model.load_checkpoint()

    # 4. Milestone 6: Train LoRA adapter with frozen base
    lora_config = config.lora
    lora_config.profiles_dir = test_profiles_dir
    personalization_mgr = PersonalizationManager(lora_config)

    trainer = PersonalizationTrainer(
        model_wrapper=model,
        personalization_mgr=personalization_mgr,
        calib_config=config.calibration,
        lora_config=lora_config,
    )

    train_res = trainer.train_on_utterances(
        user_id="test_subject_01",
        utterance_items=calib_dataset,
        epochs=2,  # Quick test epoch
        lr=1e-3,
    )

    assert train_res["trainable_parameters"] > 0
    assert Path(train_res["profile_path"]).is_file()

    # Verify base model was frozen
    for name, param in model.model.named_parameters():
        if "lora" not in name:
            assert param.requires_grad is False

    # 5. Milestone 7: Load personalized adapter independently into a fresh model
    fresh_model = AutoAVSRModel(config.model)
    fresh_model.load_checkpoint()
    fresh_mgr = PersonalizationManager(lora_config)

    loaded_meta = fresh_mgr.load_profile(
        fresh_model.model,
        user_id="test_subject_01",
        profile_dir=test_profiles_dir / "test_subject_01",
    )
    assert fresh_mgr.is_personalized is True
    assert fresh_mgr.active_mode == "Personalized"
    assert len(fresh_mgr.lora_layers) > 0

    # 6. Milestone 8 & 9: Comparative evaluation Generic vs Personalized on held-out data
    evaluator = Evaluator(fresh_model, fresh_mgr)
    report = evaluator.compare_generic_vs_personalized(
        user_id="test_subject_01",
        evaluation_items=eval_dataset,
        calibration_minutes=1.0,
        output_dir=test_experiments_dir,
    )

    assert "wer" in report
    assert "cer" in report
    assert "generic_baseline" in report
    assert "personalized" in report
    assert "inference_seconds" in report

    # Verify report saved to disk
    expected_report_file = test_experiments_dir / "eval_test_subject_01_1m.json"
    assert expected_report_file.is_file()
    with open(expected_report_file) as f:
        saved_json = json.load(f)
    assert saved_json["user"] == "test_subject_01"


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_complete_personalization_lifecycle(Path(tmp_dir))
    print("End-to-end personalization lifecycle test PASSED!")
