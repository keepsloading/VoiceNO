"""Word Error Rate (WER) and Character Error Rate (CER) calculation for VoiceNo!."""

import re
from typing import Dict, List, Optional, Union
import jiwer


def normalize_text(text: str) -> str:
    """Normalizes transcribed text for fair evaluation."""
    text = text.upper()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse consecutive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> float:
    """Computes Word Error Rate between reference and hypothesis."""
    ref_clean = normalize_text(reference)
    hyp_clean = normalize_text(hypothesis)
    if not ref_clean and not hyp_clean:
        return 0.0
    if not ref_clean:
        return 1.0
    return float(jiwer.wer(ref_clean, hyp_clean))


def compute_cer(reference: str, hypothesis: str) -> float:
    """Computes Character Error Rate between reference and hypothesis."""
    ref_clean = normalize_text(reference)
    hyp_clean = normalize_text(hypothesis)
    if not ref_clean and not hyp_clean:
        return 0.0
    if not ref_clean:
        return 1.0
    return float(jiwer.cer(ref_clean, hyp_clean))


def compute_exact_match(reference: str, hypothesis: str) -> float:
    """Computes binary exact-match accuracy between reference and hypothesis."""
    ref_clean = normalize_text(reference)
    hyp_clean = normalize_text(hypothesis)
    return 1.0 if ref_clean == hyp_clean else 0.0


def compute_topk_exact_match(reference: str, candidate_hypotheses: List[str], k: int = 3) -> float:
    """Returns 1.0 if reference exactly matches any candidate in top-k, else 0.0."""
    ref_clean = normalize_text(reference)
    for hyp in candidate_hypotheses[:k]:
        if normalize_text(hyp) == ref_clean:
            return 1.0
    return 0.0


def compute_oracle_wer(reference: str, candidate_hypotheses: List[str], k: int = 3) -> float:
    """Computes minimum WER across top-k candidate hypotheses (oracle top-k WER)."""
    if not candidate_hypotheses:
        return compute_wer(reference, "")
    wers = [compute_wer(reference, hyp) for hyp in candidate_hypotheses[:k]]
    return min(wers)


def compute_aggregate_metrics(
    references: List[str],
    hypotheses: List[str],
    nbest_hypotheses: Optional[List[List[str]]] = None,
    latencies: Optional[List[float]] = None,
    margins: Optional[List[float]] = None,
    clarifications: Optional[List[bool]] = None,
) -> Dict[str, float]:
    """Computes corpus-level WER, CER, Exact Match, Top-k accuracy, and latency."""
    assert len(references) == len(hypotheses), "References and hypotheses must be same length"

    clean_refs = [normalize_text(r) for r in references]
    clean_hyps = [normalize_text(h) for h in hypotheses]

    if not clean_refs:
        return {
            "wer": 0.0,
            "cer": 0.0,
            "exact_match": 0.0,
            "top3_accuracy": 0.0,
            "top5_accuracy": 0.0,
            "oracle_top3_wer": 0.0,
            "avg_latency": 0.0,
            "avg_margin": 0.0,
            "clarification_rate": 0.0,
        }

    total_wer = float(jiwer.wer(clean_refs, clean_hyps))
    total_cer = float(jiwer.cer(clean_refs, clean_hyps))
    exact_matches = [compute_exact_match(r, h) for r, h in zip(clean_refs, clean_hyps)]
    em_rate = float(sum(exact_matches) / len(exact_matches))

    top3_acc = 0.0
    top5_acc = 0.0
    oracle_top3_wer = total_wer

    if nbest_hypotheses and len(nbest_hypotheses) == len(references):
        top3_matches = [compute_topk_exact_match(r, cands, k=3) for r, cands in zip(clean_refs, nbest_hypotheses)]
        top5_matches = [compute_topk_exact_match(r, cands, k=5) for r, cands in zip(clean_refs, nbest_hypotheses)]
        top3_acc = float(sum(top3_matches) / len(top3_matches))
        top5_acc = float(sum(top5_matches) / len(top5_matches))
        oracle_wers = [compute_oracle_wer(r, cands, k=3) for r, cands in zip(clean_refs, nbest_hypotheses)]
        oracle_top3_wer = float(sum(oracle_wers) / len(oracle_wers))

    avg_lat = float(sum(latencies) / len(latencies)) if latencies else 0.0
    avg_margin = float(sum(margins) / len(margins)) if margins else 0.0
    clarif_rate = float(sum(1.0 for c in clarifications if c) / len(clarifications)) if clarifications else 0.0

    return {
        "wer": round(total_wer, 4),
        "cer": round(total_cer, 4),
        "exact_match": round(em_rate, 4),
        "top3_accuracy": round(top3_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "oracle_top3_wer": round(oracle_top3_wer, 4),
        "avg_latency": round(avg_lat, 4),
        "avg_margin": round(avg_margin, 4),
        "clarification_rate": round(clarif_rate, 4),
    }

