"""CLI and entry point for VoiceNo! V0.

Provides CLI commands for:
- Silent speech transcription (Generic vs Personalized)
- User calibration and LoRA training
- Held-out benchmark evaluation
- Calibration duration experiment sweeps
- Web UI launch
"""

import argparse
import json
import sys
from pathlib import Path

# Add repo root to python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voiceno.calibration.recorder import CalibrationRecorder
from voiceno.calibration.trainer import PersonalizationTrainer
from voiceno.config import VoiceNoConfig
from voiceno.evaluation.evaluate import Evaluator
from voiceno.models.auto_avsr import AutoAVSRModel
from voiceno.models.personalization import PersonalizationManager


def main():
    parser = argparse.ArgumentParser(
        prog="voiceno",
        description="VoiceNO! - Camera-only Silent Speech Recognition Prototype",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: transcribe
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe a video utterance")
    transcribe_parser.add_argument("--video", type=str, required=True, help="Path to video file")
    transcribe_parser.add_argument("--mode", type=str, choices=["generic", "personalized"], default="generic")
    transcribe_parser.add_argument("--user", type=str, default="default", help="User profile name")
    transcribe_parser.add_argument("--beam-size", type=int, default=10, help="Beam size (use 1 for fast greedy)")

    # Command: calibrate
    calibrate_parser = subparsers.add_parser("calibrate", help="Train personalized LoRA adapter")
    calibrate_parser.add_argument("--user", type=str, default="default", help="User profile name")
    calibrate_parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    calibrate_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    calibrate_parser.add_argument("--strategy", type=str, choices=["fast_cpu", "conformer_lora"], default="fast_cpu", help="Calibration strategy: fast_cpu (<50MB RAM) or conformer_lora")

    # Command: export-calib
    export_parser = subparsers.add_parser("export-calib", help="Export calibration data to zip for Colab / Cloud GPU")
    export_parser.add_argument("--user", type=str, default="default", help="User profile name")
    export_parser.add_argument("--output", type=str, default=None, help="Output zip filename")

    # Command: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Compare Generic vs Personalized on held-out data")
    eval_parser.add_argument("--user", type=str, default="default", help="User profile name")
    eval_parser.add_argument("--calib-min", type=float, default=5.0, help="Calibration duration represented")

    # Command: ui
    ui_parser = subparsers.add_parser("ui", help="Launch VoiceNO! local web interface")
    ui_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    ui_parser.add_argument("--port", type=int, default=7860, help="Port number")
    ui_parser.add_argument("--share", action="store_true", help="Create public Gradio link")

    args = parser.parse_args()

    config = VoiceNoConfig()

    if args.command == "transcribe":
        print(f"Loading Auto-AVSR model ({args.mode.capitalize()} Mode)...")
        model = AutoAVSRModel(config.model)
        model.load_checkpoint()

        mgr = PersonalizationManager(config.lora)
        if args.mode.lower() == "personalized":
            try:
                mgr.load_profile(model.model, user_id=args.user)
                print(f"Loaded personalized profile: {args.user}")
            except Exception as e:
                print(f"Warning: Could not load profile '{args.user}' ({e}). Running Generic.")

        res = model.transcribe(args.video, use_beam_search=(args.beam_size > 1))
        print("\n" + "=" * 50)
        print("Transcription Result:")
        print(f"  Text: {res['transcription']}")
        print(f"  Inference Time: {res['inference_time']}s")
        print(f"  Preprocessing Time: {res['preprocessing_time']}s")
        print(f"  Total Latency: {res['total_latency']}s")
        print("=" * 50)

    elif args.command == "calibrate":
        print(f"Starting calibration training for user '{args.user}' (Strategy: {args.strategy})...")
        recorder = CalibrationRecorder()
        dataset = recorder.load_dataset(user_id=args.user, partition="calibration")
        if not dataset:
            print(f"No calibration utterances found for user '{args.user}' in calibration_data/{args.user}/calibration.")
            sys.exit(1)

        model = AutoAVSRModel(config.model)
        model.load_checkpoint()
        trainer = PersonalizationTrainer(model, calib_config=config.calibration, lora_config=config.lora)
        res = trainer.train_on_utterances(
            user_id=args.user,
            utterance_items=dataset,
            epochs=args.epochs,
            lr=args.lr,
            strategy=args.strategy,
        )
        print("\nCalibration Complete!")
        print(json.dumps(res, indent=2))

    elif args.command == "export-calib":
        recorder = CalibrationRecorder()
        try:
            zip_path = recorder.export_calibration_package(user_id=args.user, output_zip_path=args.output)
            print(f"\nSuccessfully exported calibration package for user '{args.user}':")
            print(f"  Archive: {zip_path}")
            print(f"  Size: {zip_path.stat().st_size / 1024:.1f} KB")
            print("\nYou can upload this zip to Google Colab using 'experiments/VoiceNO_Colab_Calibration.ipynb' to train on a free GPU!")
        except Exception as e:
            print(f"Export failed: {e}")
            sys.exit(1)


    elif args.command == "evaluate":
        print(f"Running evaluation benchmark for user '{args.user}'...")
        recorder = CalibrationRecorder()
        dataset = recorder.load_dataset(user_id=args.user, partition="evaluation")
        if not dataset:
            print(f"No held-out evaluation utterances found in calibration_data/{args.user}/evaluation.")
            sys.exit(1)

        model = AutoAVSRModel(config.model)
        model.load_checkpoint()
        evaluator = Evaluator(model)
        report = evaluator.compare_generic_vs_personalized(
            user_id=args.user,
            evaluation_items=dataset,
            calibration_minutes=args.calib_min,
        )
        print("\nEvaluation Benchmark Report:")
        print(json.dumps(report, indent=2))

    elif args.command == "ui":
        from app.ui import build_ui
        print(f"Launching VoiceNO! Web UI on http://{args.host}:{args.port}...")
        ui = build_ui(config=config)
        ui.launch(server_name=args.host, server_port=args.port, share=args.share)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
