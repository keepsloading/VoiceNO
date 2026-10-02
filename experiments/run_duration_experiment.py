"""Calibration-duration experiment runner for VoiceNo! V0.

Evaluates how visual speech recognition performance scales with calibration duration:
[0m, 1m, 5m, 15m, 30m, 45m].
Saves machine-readable experimental artifacts and calibration curves.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voiceno.calibration.recorder import CalibrationRecorder
from voiceno.calibration.trainer import PersonalizationTrainer
from voiceno.config import REPO_ROOT, VoiceNoConfig
from voiceno.evaluation.evaluate import Evaluator
from voiceno.models.auto_avsr import AutoAVSRModel
from voiceno.models.personalization import PersonalizationManager


def run_experiment(
    user_id: str = "default",
    durations_minutes: Optional[List[float]] = None,
    epochs_per_tier: int = 5,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Executes the calibration duration sweep experiment."""
    if durations_minutes is None:
        durations_minutes = [0.0, 1.0, 5.0, 15.0, 30.0, 45.0]

    output_dir = output_dir or (REPO_ROOT / "experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = VoiceNoConfig()
    recorder = CalibrationRecorder()

    calib_items = recorder.load_dataset(user_id=user_id, partition="calibration")
    eval_items = recorder.load_dataset(user_id=user_id, partition="evaluation")

    if not eval_items:
        print(f"Error: No held-out evaluation utterances found for user '{user_id}'.")
        print(f"Please record evaluation samples in calibration_data/{user_id}/evaluation first.")
        return {}

    print("=" * 65)
    print(f"Running VoiceNo! Calibration-Duration Scaling Experiment")
    print(f"User: {user_id}")
    print(f"Held-out Evaluation Utterances: {len(eval_items)}")
    print(f"Duration Tiers (minutes): {durations_minutes}")
    print("=" * 65)

    model = AutoAVSRModel(config.model)
    model.load_checkpoint()

    evaluator = Evaluator(model)
    results = []

    for dur_min in durations_minutes:
        dur_sec = dur_min * 60.0
        print(f"\n--- Testing Calibration Tier: {dur_min} min ---")

        if dur_min == 0.0:
            # 0 minutes = Generic baseline (no adaptation)
            eval_res = evaluator.evaluate_utterances(eval_items, mode="Generic")
            tier_record = {
                "calibration_minutes": 0.0,
                "mode": "Generic Baseline",
                "wer": eval_res["wer"],
                "cer": eval_res["cer"],
                "avg_inference_seconds": eval_res["avg_inference_seconds"],
                "num_calibration_utterances": 0,
            }
        else:
            if not calib_items:
                print(f"Warning: No calibration items available. Skipping {dur_min}m tier.")
                continue

            tier_mgr = PersonalizationManager(config.lora)
            trainer = PersonalizationTrainer(
                model_wrapper=model,
                personalization_mgr=tier_mgr,
                calib_config=config.calibration,
                lora_config=config.lora,
            )

            tier_user_id = f"{user_id}_{int(dur_min)}m"
            train_res = trainer.train_on_utterances(
                user_id=tier_user_id,
                utterance_items=calib_items,
                max_duration_seconds=dur_sec,
                epochs=epochs_per_tier,
            )

            # Evaluate personalized model
            eval_res = evaluator.evaluate_utterances(eval_items, mode="Personalized")
            tier_record = {
                "calibration_minutes": dur_min,
                "actual_calibration_minutes": train_res["calibration_minutes"],
                "mode": "Personalized (LoRA)",
                "wer": eval_res["wer"],
                "cer": eval_res["cer"],
                "avg_inference_seconds": eval_res["avg_inference_seconds"],
                "trainable_parameters": train_res["trainable_parameters"],
                "training_seconds": train_res["training_seconds"],
                "num_calibration_utterances": train_res["num_calibration_utterances"],
            }

        results.append(tier_record)
        print(f"  WER: {tier_record['wer'] * 100:.2f}% | CER: {tier_record['cer'] * 100:.2f}%")

    experiment_report = {
        "user": user_id,
        "experiment_name": "calibration_duration_scaling",
        "timestamp": int(time.time()),
        "checkpoint": str(config.model.checkpoint_path),
        "tiers": results,
    }

    report_path = output_dir / f"calibration_curve_{user_id}.json"
    with open(report_path, "w") as f:
        json.dump(experiment_report, f, indent=2)

    print("\n" + "=" * 65)
    print("CALIBRATION DURATION SWEEP SUMMARY")
    print(f"{'Duration (min)':<18} {'Mode':<22} {'WER (%)':<10} {'CER (%)':<10}")
    print("-" * 65)
    for r in results:
        print(f"{r['calibration_minutes']:<18} {r['mode']:<22} {r['wer']*100:<10.2f} {r['cer']*100:<10.2f}")
    print("=" * 65)
    print(f"Full report saved to: {report_path}")

    return experiment_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoiceNo! Calibration-Duration Scaling Experiment")
    parser.add_argument("--user", type=str, default="default", help="User profile ID")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per tier")
    args = parser.parse_args()

    run_experiment(user_id=args.user, epochs_per_tier=args.epochs)
