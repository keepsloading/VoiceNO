"""Evaluation engine comparing Generic Auto-AVSR vs Personalized Memact.

Evaluates supplied utterances and records actual predictions/model state.
Speaker independence and label accuracy require a separately verified dataset.
"""

import json
import time
import copy
import hashlib
import platform
from importlib.metadata import PackageNotFoundError, version
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch

from memact.config import REPO_ROOT
from memact.context.rescorer import ContextRescorer, TaskContext
from memact.evaluation.metrics import (
    compute_aggregate_metrics,
    compute_cer,
    compute_exact_match,
    compute_topk_exact_match,
    compute_wer,
)
from memact.models.auto_avsr import AutoAVSRModel
from memact.models.personalization import LoRALinear, PersonalizationManager
from memact.provenance import code_revision, file_sha256, manifest_sha256


class Evaluator:
    """Evaluates generic vs personalized recognition performance on held-out utterances."""

    def __init__(
        self,
        model_wrapper: AutoAVSRModel,
        personalization_mgr: Optional[PersonalizationManager] = None,
        context_rescorer: Optional[ContextRescorer] = None,
    ):
        self.model_wrapper = model_wrapper
        self.personalization_mgr = personalization_mgr or PersonalizationManager()
        self.context_rescorer = context_rescorer or ContextRescorer()

    def _recognition_state(self, mode: str) -> Dict:
        """Reject report labels that disagree with the model's actual adapters."""
        if mode not in ("Generic", "Personalized"):
            raise ValueError("mode must be Generic or Personalized")
        model = self.model_wrapper.model
        if model is None:
            raise RuntimeError("Load a checkpoint before evaluation")
        if hasattr(self.model_wrapper, "checkpoint_info") and not self.model_wrapper.checkpoint_info:
            raise RuntimeError("Evaluation requires a loaded checkpoint, not randomly initialized weights")
        actual = {name: layer for name, layer in model.named_modules() if isinstance(layer, LoRALinear)}
        if mode == "Generic" and actual:
            raise ValueError("Generic evaluation requested while LoRA adapters are active")
        if mode == "Personalized":
            managed = self.personalization_mgr.lora_layers
            if not actual or not self.personalization_mgr.is_personalized or actual != managed:
                raise ValueError("Personalized evaluation requires active adapters owned by this manager")
        digest = hashlib.sha256()
        for name, layer in sorted(actual.items()):
            digest.update(name.encode())
            digest.update(str(layer.scaling).encode())
            for tensor in (layer.lora_A, layer.lora_B):
                digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return {
            "mode": mode,
            "adapted_layers": sorted(actual),
            "adapter_sha256": digest.hexdigest() if actual else None,
            "user_id": self.personalization_mgr.active_user_id if actual else None,
            "profile": copy.deepcopy(self.personalization_mgr.profile_info) if actual else None,
            "checkpoint": copy.deepcopy(getattr(self.model_wrapper, "checkpoint_info", None)),
        }

    @staticmethod
    def _validate_items(items: List[Dict]) -> None:
        if not items:
            raise ValueError("No evaluation recordings supplied; an empty run is not a benchmark")
        ids = [item.get("utterance_id", Path(item["video_path"]).stem) for item in items]
        if len(set(ids)) != len(ids):
            raise ValueError("Evaluation utterance IDs must be unique")
        if any(not isinstance(item.get("prompt"), str) for item in items):
            raise ValueError("Every evaluation recording needs a reference prompt string")

    def _provenance(self, items: List[Dict]) -> Dict:
        manifest = []
        for item in items:
            entry = copy.deepcopy(item)
            path = Path(item["video_path"])
            entry["video_sha256"] = file_sha256(path) if path.is_file() else None
            manifest.append(entry)
        config = getattr(self.model_wrapper, "config", None)
        preprocessor = getattr(self.model_wrapper, "preprocessor", None)
        prep_config = getattr(preprocessor, "config", None)
        packages = {}
        for name in ("torch", "torchvision", "mediapipe", "opencv-python", "jiwer", "sentencepiece"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        return {
            "runtime": {"python": platform.python_version(), "packages": packages},
            "code": code_revision(REPO_ROOT),
            "checkpoint": copy.deepcopy(getattr(self.model_wrapper, "checkpoint_info", None)),
            "model_settings": json.loads(json.dumps(asdict(config), default=str)) if config else None,
            "preprocessing_settings": json.loads(json.dumps(asdict(prep_config), default=str)) if prep_config else None,
            "dataset": manifest,
            "dataset_sha256": manifest_sha256(manifest),
            "limitations": [
                "Reference labels and silent articulation require human verification.",
                "Separate folders alone do not establish speaker/session independence.",
                "Model confidence values are uncalibrated candidate scores.",
            ],
        }

    def evaluate_utterances(
        self,
        evaluation_items: List[Dict],
        mode: str = "Generic",
        use_beam_search: bool = True,
        apply_context: bool = False,
        visual_results: Optional[Dict] = None,
    ) -> Dict:
        """Evaluates held-out utterance items under a specific mode.

        Supports:
        - Mode: 'Generic' or 'Personalized'
        - apply_context: Reranks N-best candidates with ContextRescorer if True
        """
        self._validate_items(evaluation_items)
        state = self._recognition_state(mode)
        self.model_wrapper.model.eval()
        if visual_results is not None:
            if not apply_context or visual_results["recognition_state"] != state:
                raise ValueError("Cached visual results must match the active model and be used for rescoring")
            if visual_results["evaluation_items_sha256"] != manifest_sha256(evaluation_items):
                raise ValueError("Cached visual results do not match these evaluation items")
            if visual_results["applied_context"]:
                raise ValueError("Cached results must contain unrescored visual hypotheses")
            if visual_results["use_beam_search"] != use_beam_search:
                raise ValueError("Cached results use different decoder settings")
            config = getattr(self.model_wrapper, "config", None)
            if visual_results["decoder_settings"] != {
                "beam_size": getattr(config, "beam_size", None),
                "ctc_weight": getattr(config, "ctc_weight", None),
            }:
                raise ValueError("Cached results use different decoder settings")
        references = []
        hypotheses = []
        nbest_list = []
        latencies = []
        margins = []
        clarifications = []
        per_utterance_results = []

        for index, item in enumerate(evaluation_items):
            v_path = item["video_path"]
            ground_truth = item["prompt"]

            started = time.perf_counter()
            status, error = "ok", None
            if visual_results is None:
                try:
                    res = self.model_wrapper.transcribe(v_path, use_beam_search=use_beam_search)
                    cand_hyps = copy.deepcopy(res.get("hypotheses", [{"text": res["transcription"], "score": 0.0}]))
                except Exception as exc:
                    # A failed recording remains in the denominator as an empty
                    # prediction, with its error recorded instead of disappearing.
                    status, error = "error", f"{type(exc).__name__}: {exc}"
                    res = {"transcription": "", "inference_time": 0.0,
                           "total_latency": time.perf_counter() - started,
                           "confidence_level": "UNAVAILABLE", "needs_clarification": True}
                    cand_hyps = []
            else:
                cached = visual_results["utterances"][index]
                res = copy.deepcopy(cached["raw_result"])
                cand_hyps = copy.deepcopy(cached["visual_hypotheses"])
                status, error = cached["status"], cached["error"]

            visual_hypotheses = copy.deepcopy(cand_hyps)
            ranking_started = time.perf_counter()

            if apply_context and status == "ok":
                ctx_meta = item.get("task_context")
                task_ctx = TaskContext(
                    active_app=ctx_meta.get("active_app") if isinstance(ctx_meta, dict) else None,
                    selected_text=ctx_meta.get("selected_text") if isinstance(ctx_meta, dict) else None,
                    preceding_text=ctx_meta.get("preceding_text") if isinstance(ctx_meta, dict) else None,
                    domain_keywords=ctx_meta.get("domain_keywords", []) if isinstance(ctx_meta, dict) else [],
                ) if ctx_meta else None
                cand_hyps = self.context_rescorer.rescore_hypotheses(cand_hyps, context=task_ctx)

            ranking_seconds = time.perf_counter() - ranking_started if apply_context else 0.0

            top_hyp = cand_hyps[0]["text"] if cand_hyps else res["transcription"]
            cand_texts = [h["text"] for h in cand_hyps]

            wer_score = compute_wer(ground_truth, top_hyp)
            cer_score = compute_cer(ground_truth, top_hyp)
            em_score = compute_exact_match(ground_truth, top_hyp) if status == "ok" else 0.0
            top3_hit = compute_topk_exact_match(ground_truth, cand_texts, k=3) if status == "ok" else 0.0

            references.append(ground_truth)
            hypotheses.append(top_hyp)
            nbest_list.append(cand_texts)
            latencies.append(res["inference_time"])
            margins.append(res.get("score_margin", 0.0))
            clarifications.append(res.get("needs_clarification", False))

            per_utterance_results.append({
                "utterance_id": item.get("utterance_id", Path(v_path).stem),
                "reference": ground_truth,
                "hypothesis": top_hyp,
                "nbest": cand_texts[:5],
                "candidates": cand_hyps,
                "visual_hypotheses": visual_hypotheses,
                "raw_result": res,
                "status": status,
                "error": error,
                "wer": round(wer_score, 4),
                "cer": round(cer_score, 4),
                "exact_match": em_score,
                "in_top3": bool(top3_hit),
                "confidence_level": res.get("confidence_level", "HIGH"),
                "score_margin": res.get("score_margin", 0.0),
                "inference_seconds": res["inference_time"],
                "total_latency": res["total_latency"],
                "ranking_seconds": ranking_seconds,
                "recognition_plus_ranking_seconds": res["total_latency"] + ranking_seconds,
            })

        aggregate = compute_aggregate_metrics(
            references=references,
            hypotheses=hypotheses,
            nbest_hypotheses=nbest_list,
            latencies=latencies,
            margins=margins,
            clarifications=clarifications,
        )
        # A failed negative/no-speech clip must not count as a correct empty output.
        aggregate["exact_match"] = round(sum(row["exact_match"] for row in per_utterance_results) / len(evaluation_items), 4)

        return {
            "mode": mode,
            "recognition_state": state,
            "evaluation_items_sha256": manifest_sha256(evaluation_items),
            "use_beam_search": use_beam_search,
            "decoder_settings": {
                "beam_size": getattr(getattr(self.model_wrapper, "config", None), "beam_size", None),
                "ctc_weight": getattr(getattr(self.model_wrapper, "config", None), "ctc_weight", None),
            },
            "visual_results_reused": visual_results is not None,
            "num_failed": sum(row["status"] == "error" for row in per_utterance_results),
            "applied_context": apply_context,
            "num_utterances": len(evaluation_items),
            "wer": aggregate["wer"],
            "cer": aggregate["cer"],
            "exact_match": aggregate["exact_match"],
            "top3_accuracy": aggregate["top3_accuracy"],
            "top5_accuracy": aggregate["top5_accuracy"],
            "oracle_top3_wer": aggregate["oracle_top3_wer"],
            "avg_inference_seconds": aggregate["avg_latency"],
            "avg_score_margin": aggregate["avg_margin"],
            "clarification_rate": aggregate["clarification_rate"],
            "uncertainty_metrics_source": "raw visual model; not recalibrated after ranking",
            "utterances": per_utterance_results,
        }

    def evaluate_four_conditions(
        self,
        user_id: str,
        evaluation_items: List[Dict],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Runs side-by-side evaluation across all 4 experimental conditions:

        Condition A: Generic open-vocabulary recognition
        Condition B: Personalized open-vocabulary recognition
        Condition C: Personalized + N-best hypotheses (Oracle Top-3/Top-5)
        Condition D: Personalized + N-best + Contextual rescoring
        """
        output_dir = Path(output_dir or (REPO_ROOT / "experiments"))
        output_dir.mkdir(parents=True, exist_ok=True)

        self._validate_items(evaluation_items)
        provenance = self._provenance(evaluation_items)
        # Always remove physical adapters, even if a stale manager says Generic.
        self.personalization_mgr.remove_lora(self.model_wrapper.model)
        cond_A = self.evaluate_utterances(evaluation_items, mode="Generic", apply_context=False)

        # 2. Condition B & C: Personalized
        self.personalization_mgr.load_profile(self.model_wrapper.model, user_id=user_id)
        cond_B = self.evaluate_utterances(evaluation_items, mode="Personalized", apply_context=False)

        # 3. Condition D: Personalized + Context Rescoring
        cond_D = self.evaluate_utterances(
            evaluation_items, mode="Personalized", apply_context=True, visual_results=cond_B,
        )

        from memact.evaluation.analysis import analyze_predictions

        report = {
            "schema_version": 2,
            "provenance": provenance,
            "user_id": user_id,
            "timestamp": int(time.time()),
            "num_evaluation_utterances": len(evaluation_items),
            "conditions": {
                "A_generic_baseline": cond_A,
                "B_personalized": cond_B,
                "C_personalized_nbest_oracle": {
                    "description": "Personalized + N-best Hypotheses (Oracle Top-3/Top-5)",
                    "top3_accuracy": cond_B["top3_accuracy"],
                    "top5_accuracy": cond_B["top5_accuracy"],
                    "oracle_top3_wer": cond_B["oracle_top3_wer"],
                    "oracle_only": True,
                    "note": "Uses reference labels to choose candidates; not an achievable decoding result.",
                },
                "D_personalized_context": cond_D,
            },
            "failure_analysis": analyze_predictions(cond_A, cond_B, cond_D),
            "ranking": {
                "method": "heuristic n-gram/domain rescoring" if not self.context_rescorer.use_neural_lm else "neural LM with heuristic domain bias and statistical fallback",
                "weight_lm": self.context_rescorer.weight_lm,
                "domain": self.context_rescorer.domain.name if self.context_rescorer.domain else None,
                "expands_candidates_with_handwritten_rules": True,
            },
            "deltas": {
                "personalization_wer_reduction": round(cond_A["wer"] - cond_B["wer"], 4),
                "context_wer_reduction": round(cond_B["wer"] - cond_D["wer"], 4),
                "nbest_oracle_gain": round(cond_B["wer"] - cond_B["oracle_top3_wer"], 4),
            },
        }

        save_path = output_dir / f"open_vocab_eval_{user_id}.json"
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)

        return report

    def compare_generic_vs_personalized(
        self,
        user_id: str,
        evaluation_items: List[Dict],
        calibration_minutes: float = 5.0,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict:
        """Runs comparative evaluation of Generic vs Personalized models (compatible with existing tests)."""
        output_dir = Path(output_dir or (REPO_ROOT / "experiments"))
        output_dir.mkdir(parents=True, exist_ok=True)

        self._validate_items(evaluation_items)
        provenance = self._provenance(evaluation_items)
        self.personalization_mgr.remove_lora(self.model_wrapper.model)
        generic_results = self.evaluate_utterances(evaluation_items, mode="Generic")

        self.personalization_mgr.load_profile(self.model_wrapper.model, user_id=user_id)
        personalized_results = self.evaluate_utterances(evaluation_items, mode="Personalized")

        wer_delta = generic_results["wer"] - personalized_results["wer"]
        cer_delta = generic_results["cer"] - personalized_results["cer"]

        report = {
            "schema_version": 2,
            "provenance": provenance,
            "user": user_id,
            "calibration_minutes": calibration_minutes,
            "wer": personalized_results["wer"],
            "cer": personalized_results["cer"],
            "exact_match": personalized_results.get("exact_match", 0.0),
            "top3_accuracy": personalized_results.get("top3_accuracy", 0.0),
            "inference_seconds": personalized_results["avg_inference_seconds"],
            "generic_baseline": {
                "recognition_state": generic_results["recognition_state"],
                "num_failed": generic_results["num_failed"],
                "utterances": generic_results["utterances"],
                "wer": generic_results["wer"],
                "cer": generic_results["cer"],
                "exact_match": generic_results.get("exact_match", 0.0),
                "inference_seconds": generic_results["avg_inference_seconds"],
            },
            "personalized": {
                "recognition_state": personalized_results["recognition_state"],
                "num_failed": personalized_results["num_failed"],
                "utterances": personalized_results["utterances"],
                "wer": personalized_results["wer"],
                "cer": personalized_results["cer"],
                "exact_match": personalized_results.get("exact_match", 0.0),
                "top3_accuracy": personalized_results.get("top3_accuracy", 0.0),
                "inference_seconds": personalized_results["avg_inference_seconds"],
            },
            "absolute_improvements": {
                "wer_reduction": round(wer_delta, 4),
                "cer_reduction": round(cer_delta, 4),
            },
            "timestamp": int(time.time()),
        }

        save_path = output_dir / f"eval_{user_id}_{int(calibration_minutes)}m.json"
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)

        return report
