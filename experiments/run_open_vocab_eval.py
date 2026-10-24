"""Evaluate real held-out recordings on Colab; CPU inference requires opt-in.

Baseline: python experiments/run_open_vocab_eval.py --user default
Comparison: add --personalized after training and importing a user profile.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voiceno.calibration.recorder import CalibrationRecorder
from voiceno.config import VoiceNoConfig
from voiceno.evaluation.evaluate import Evaluator
from voiceno.provenance import file_sha256


def build_evaluation_items(
    user_id: str, data_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Use only user evaluation recordings; never pad with guessed sample labels."""
    recorder = CalibrationRecorder(base_dir=data_dir)
    items = recorder.load_dataset(user_id=user_id, partition="evaluation")
    Evaluator._validate_items(items)
    calibration_hashes = {
        file_sha256(item["video_path"])
        for item in recorder.load_dataset(user_id=user_id, partition="calibration")
        if Path(item["video_path"]).is_file()
    }
    for item in items:
        path = Path(item["video_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation recording: {path}")
        if file_sha256(path) in calibration_hashes:
            raise ValueError(f"Recording appears in both calibration and evaluation: {path.name}")
    return items


def run_open_vocab_evaluation(
    user_id: str = "default",
    output_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    profiles_dir: Optional[Path] = None,
    personalized: bool = False,
    device: str = "cuda",
    beam_size: int = 10,
):
    """Run a real baseline or the shared evaluator's four-condition comparison."""
    if beam_size < 1:
        raise ValueError("beam_size must be positive")
    if personalized and beam_size == 1:
        raise ValueError("The N-best comparison requires beam_size > 1")
    # Validate recordings/profile before loading the expensive model.
    items = build_evaluation_items(user_id, data_dir=data_dir)
    config = VoiceNoConfig()
    if profiles_dir is not None:
        config.lora.profiles_dir = Path(profiles_dir)
    if personalized and not (config.lora.profiles_dir / user_id / "lora.pt").is_file():
        raise FileNotFoundError("No personalized profile found. Run the baseline first or import a trained profile.")

    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "A GPU is required by default. Run this command in a Colab GPU notebook; "
            "--device cpu explicitly opts into heavy local inference."
        )
    from voiceno.models.auto_avsr import AutoAVSRModel
    from voiceno.models.personalization import PersonalizationManager

    config.model.device = device
    config.model.beam_size = beam_size
    output_dir = Path(output_dir or REPO_ROOT / "experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = AutoAVSRModel(config.model)
    print(f"Loading Auto-AVSR on {device}; {len(items)} held-out recordings.")
    model.load_checkpoint()
    evaluator = Evaluator(model, PersonalizationManager(config.lora))
    if personalized:
        report = evaluator.evaluate_four_conditions(user_id, items, output_dir=output_dir)
        conditions = report["conditions"]
    else:
        baseline = evaluator.evaluate_utterances(items, use_beam_search=beam_size > 1)
        report = {
            "schema_version": 2,
            "user_id": user_id,
            "timestamp": int(time.time()),
            "provenance": evaluator._provenance(items),
            "num_evaluation_utterances": len(items),
            "conditions": {"A_generic_baseline": baseline},
        }
        conditions = report["conditions"]
    report_path = output_dir / f"open_vocab_benchmark_{user_id}.json"
    report_path.write_text(json.dumps(report, indent=2))
    for name, result in conditions.items():
        if result.get("oracle_only"):
            print(f"{name}: oracle top-3 WER={result['oracle_top3_wer']:.1%} (reference-assisted ceiling)")
        else:
            print(f"{name}: WER={result['wer']:.1%}, exact match={result['exact_match']:.1%}, failures={result['num_failed']}")
    print(f"Actual predictions and run provenance saved to {report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default="default")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--profiles-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--personalized", action="store_true", help="Compare a saved profile against the generic baseline")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--beam-size", type=int, default=10)
    args = parser.parse_args()
    try:
        run_open_vocab_evaluation(
            user_id=args.user, output_dir=args.output_dir, data_dir=args.data_dir,
            profiles_dir=args.profiles_dir, personalized=args.personalized,
            device=args.device, beam_size=args.beam_size,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(1, f"Evaluation stopped: {exc}\n")
