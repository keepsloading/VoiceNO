"""Evaluation engine comparing Generic Auto-AVSR vs Personalized VoiceNo.

Strictly evaluates on held-out utterances never seen during calibration.
Generates machine-readable benchmark reports and calibration duration experiments.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from voiceno.config import REPO_ROOT
from voiceno.context.rescorer import ContextRescorer, TaskContext
from voiceno.evaluation.metrics import (
    compute_aggregate_metrics,
    compute_cer,
    compute_exact_match,
    compute_oracle_wer,
    compute_topk_exact_match,
    compute_wer,
)
from voiceno.models.auto_avsr import AutoAVSRModel
from voiceno.models.personalization import PersonalizationManager


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

    def evaluate_utterances(
        self,
        evaluation_items: List[Dict],
        mode: str = "Generic",
        use_beam_search: bool = True,
        apply_context: bool = False,
    ) -> Dict:
        """Evaluates held-out utterance items under a specific mode.

        Supports:
        - Mode: 'Generic' or 'Personalized'
        - apply_context: Reranks N-best candidates with ContextRescorer if True
        """
        references = []
        hypotheses = []
        nbest_list = []
        latencies = []
        margins = []
        clarifications = []
        per_utterance_results = []

        for item in evaluation_items:
            v_path = item["video_path"]
            ground_truth = item["prompt"]

            res = self.model_wrapper.transcribe(v_path, use_beam_search=use_beam_search)
            cand_hyps = res.get("hypotheses", [{"text": res["transcription"], "score": 0.0}])

            if apply_context:
                ctx_meta = item.get("task_context")
                task_ctx = TaskContext(
                    active_app=ctx_meta.get("active_app") if isinstance(ctx_meta, dict) else None,
                    selected_text=ctx_meta.get("selected_text") if isinstance(ctx_meta, dict) else None,
                    preceding_text=ctx_meta.get("preceding_text") if isinstance(ctx_meta, dict) else None,
                    domain_keywords=ctx_meta.get("domain_keywords", []) if isinstance(ctx_meta, dict) else [],
                ) if ctx_meta else None
                cand_hyps = self.context_rescorer.rescore_hypotheses(cand_hyps, context=task_ctx)

            top_hyp = cand_hyps[0]["text"] if cand_hyps else res["transcription"]
            cand_texts = [h["text"] for h in cand_hyps]

            wer_score = compute_wer(ground_truth, top_hyp)
            cer_score = compute_cer(ground_truth, top_hyp)
            em_score = compute_exact_match(ground_truth, top_hyp)
            top3_hit = compute_topk_exact_match(ground_truth, cand_texts, k=3)

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
                "wer": round(wer_score, 4),
                "cer": round(cer_score, 4),
                "exact_match": em_score,
                "in_top3": bool(top3_hit),
                "confidence_level": res.get("confidence_level", "HIGH"),
                "score_margin": res.get("score_margin", 0.0),
                "inference_seconds": res["inference_time"],
                "total_latency": res["total_latency"],
            })

        aggregate = compute_aggregate_metrics(
            references=references,
            hypotheses=hypotheses,
            nbest_hypotheses=nbest_list,
            latencies=latencies,
            margins=margins,
            clarifications=clarifications,
        )

        return {
            "mode": mode,
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

        # 1. Condition A: Generic Baseline
        if self.personalization_mgr.is_personalized:
            self.personalization_mgr.remove_lora(self.model_wrapper.model)
        cond_A = self.evaluate_utterances(evaluation_items, mode="Generic", apply_context=False)

        # 2. Condition B & C: Personalized
        self.personalization_mgr.load_profile(self.model_wrapper.model, user_id=user_id)
        cond_B = self.evaluate_utterances(evaluation_items, mode="Personalized", apply_context=False)

        # 3. Condition D: Personalized + Context Rescoring
        cond_D = self.evaluate_utterances(evaluation_items, mode="Personalized", apply_context=True)

        report = {
            "user_id": user_id,
            "timestamp": int(time.time()),
            "num_evaluation_utterances": len(evaluation_items),
            "conditions": {
                "A_generic_baseline": {
                    "description": "Generic open-vocabulary recognition (Top-1)",
                    "wer": cond_A["wer"],
                    "cer": cond_A["cer"],
                    "exact_match": cond_A["exact_match"],
                    "avg_inference_seconds": cond_A["avg_inference_seconds"],
                },
                "B_personalized": {
                    "description": "Personalized open-vocabulary recognition (Top-1)",
                    "wer": cond_B["wer"],
                    "cer": cond_B["cer"],
                    "exact_match": cond_B["exact_match"],
                    "avg_inference_seconds": cond_B["avg_inference_seconds"],
                },
                "C_personalized_nbest_oracle": {
                    "description": "Personalized + N-best Hypotheses (Oracle Top-3/Top-5)",
                    "top3_accuracy": cond_B["top3_accuracy"],
                    "top5_accuracy": cond_B["top5_accuracy"],
                    "oracle_top3_wer": cond_B["oracle_top3_wer"],
                },
                "D_personalized_context": {
                    "description": "Personalized + N-best + Contextual Rescoring",
                    "wer": cond_D["wer"],
                    "cer": cond_D["cer"],
                    "exact_match": cond_D["exact_match"],
                    "top3_accuracy": cond_D["top3_accuracy"],
                    "avg_inference_seconds": cond_D["avg_inference_seconds"],
                },
            },
            "delts": {
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

        if self.personalization_mgr.is_personalized:
            self.personalization_mgr.remove_lora(self.model_wrapper.model)
        generic_results = self.evaluate_utterances(evaluation_items, mode="Generic")

        self.personalization_mgr.load_profile(self.model_wrapper.model, user_id=user_id)
        personalized_results = self.evaluate_utterances(evaluation_items, mode="Personalized")

        wer_delta = generic_results["wer"] - personalized_results["wer"]
        cer_delta = generic_results["cer"] - personalized_results["cer"]

        report = {
            "user": user_id,
            "calibration_minutes": calibration_minutes,
            "wer": personalized_results["wer"],
            "cer": personalized_results["cer"],
            "exact_match": personalized_results.get("exact_match", 0.0),
            "top3_accuracy": personalized_results.get("top3_accuracy", 0.0),
            "inference_seconds": personalized_results["avg_inference_seconds"],
            "generic_baseline": {
                "wer": generic_results["wer"],
                "cer": generic_results["cer"],
                "exact_match": generic_results.get("exact_match", 0.0),
                "inference_seconds": generic_results["avg_inference_seconds"],
            },
            "personalized": {
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

