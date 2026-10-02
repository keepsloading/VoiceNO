"""Standalone cloud/Colab training script for VoiceNO!

Allows training full Conformer LoRA on a free Google Colab / GPU machine
and exporting the lightweight lora.pt back to your local laptop.

Usage:
    python experiments/colab_train.py --zip default_calibration_package.zip --user default --epochs 15
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import time
import zipfile
import torch

from voiceno.config import REPO_ROOT, VoiceNoConfig
from voiceno.models.auto_avsr import AutoAVSRModel
from voiceno.models.personalization import PersonalizationManager
from voiceno.calibration.trainer import PersonalizationTrainer
from voiceno.calibration.recorder import CalibrationRecorder
from voiceno.evaluation.evaluate import Evaluator


def main():
    parser = argparse.ArgumentParser(description="VoiceNO! Cloud/Colab LoRA Fine-tuning")
    parser.add_argument("--zip", type=str, required=True, help="Path to exported calibration_package.zip")
    parser.add_argument("--user", type=str, default="default", help="User profile ID")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--strategy", type=str, default="conformer_lora", choices=["conformer_lora", "fast_cpu"])
    parser.add_argument("--output_dir", type=str, default="colab_output", help="Directory to save output lora.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== VoiceNO! Personalization Trainer ===")
    print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Device: {device}")
    print(f"User: {args.user}")
    print(f"Strategy: {args.strategy}")

    # 1. Unzip calibration package
    extract_dir = Path("extracted_calib") / args.user
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip, "r") as zipf:
        zipf.extractall(extract_dir)
    print(f"Extracted calibration package to: {extract_dir}")

    recorder = CalibrationRecorder(base_dir=Path("extracted_calib"))
    calib_items = recorder.load_dataset(user_id=args.user, partition="calibration")
    eval_items = recorder.load_dataset(user_id=args.user, partition="evaluation")
    print(f"Loaded {len(calib_items)} calibration utterance(s) and {len(eval_items)} evaluation utterance(s).")

    if not calib_items:
        raise ValueError("No calibration items found in the provided zip package!")

    # 2. Load model
    config = VoiceNoConfig()
    config.model.device = device
    model_wrapper = AutoAVSRModel(config.model)
    print("Loading Auto-AVSR base checkpoint...")
    model_wrapper.load_checkpoint()

    # 3. Train
    out_dir = Path(args.output_dir) / args.user
    out_dir.mkdir(parents=True, exist_ok=True)
    config.lora.profiles_dir = Path(args.output_dir)

    pm = PersonalizationManager(config.lora)
    trainer = PersonalizationTrainer(
        model_wrapper=model_wrapper,
        personalization_mgr=pm,
        calib_config=config.calibration,
        lora_config=config.lora,
    )

    t0 = time.time()
    res = trainer.train_on_utterances(
        user_id=args.user,
        utterance_items=calib_items,
        epochs=args.epochs,
        lr=args.lr,
        strategy=args.strategy,
    )
    t_train = time.time() - t0
    print(f"\nTraining completed in {t_train:.2f}s!")
    print(f"Trainable parameters: {res['trainable_parameters']:,}")
    print(f"Profile saved to: {res['profile_path']}")

    # 4. Evaluate if eval items available
    if eval_items:
        print("\nEvaluating Baseline vs Personalized on held-out sentences...")
        evaluator = Evaluator(model_wrapper, pm)
        report = evaluator.compare_generic_vs_personalized(
            user_id=args.user,
            evaluation_items=eval_items,
        )
        print("Evaluation Report:")
        print(json.dumps(report, indent=2))
        with open(out_dir / "eval_report.json", "w") as f:
            json.dump(report, f, indent=2)

    # 5. Zip result for easy download
    result_zip = Path(f"{args.user}_personalized_weights.zip")
    with zipfile.ZipFile(str(result_zip), "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in out_dir.iterdir():
            if f.is_file():
                zipf.write(f, f.name)
    print(f"\nCreated download package: {result_zip.resolve()} ({result_zip.stat().st_size / 1024:.1f} KB)")
    print(f"Drop 'lora.pt' and 'metadata.json' into user_profiles/{args.user}/ on your laptop.")


if __name__ == "__main__":
    main()
