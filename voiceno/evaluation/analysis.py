"""Describe observed prediction changes without inventing causal explanations."""

from voiceno.evaluation.metrics import normalize_text


def analyze_predictions(generic, personalized, contextual):
    """Return one evidence row per recording, including failures and regressions.

    These categories describe outcomes. Predictions alone cannot prove that an
    error was caused by an intrinsically indistinguishable viseme or anatomy.
    """
    groups = [result["utterances"] for result in (generic, personalized, contextual)]
    identifiers = [[row["utterance_id"] for row in rows] for rows in groups]
    if identifiers[0] != identifiers[1] or identifiers[0] != identifiers[2]:
        raise ValueError("Conditions must contain the same recordings in the same order")
    cases = []
    for base, adapted, ranked in zip(*groups):
        if len({row["reference"] for row in (base, adapted, ranked)}) != 1:
            raise ValueError("Condition references disagree")
        categories = []
        if any(row["status"] == "error" for row in (base, adapted, ranked)):
            categories.append("processing_failure")
        if base["exact_match"]:
            categories.append("generic_correct")
        if not adapted["exact_match"] and any(
            normalize_text(text) == normalize_text(adapted["reference"]) for text in adapted["nbest"]
        ):
            categories.append("correct_in_personalized_top5")
        if adapted["wer"] < base["wer"]:
            categories.append("personalization_improved")
        elif adapted["wer"] > base["wer"]:
            categories.append("personalization_regressed")
        if ranked["wer"] < adapted["wer"]:
            categories.append("rescoring_improved")
        elif ranked["wer"] > adapted["wer"]:
            categories.append("rescoring_regressed")
        if not ranked["exact_match"]:
            categories.append("unresolved_error")
        cases.append({
            "utterance_id": base["utterance_id"],
            "reference": base["reference"],
            "generic_hypothesis": base["hypothesis"],
            "personalized_hypothesis": adapted["hypothesis"],
            "rescored_hypothesis": ranked["hypothesis"],
            "personalized_nbest": adapted["nbest"],
            "categories": categories,
            "evidence_source": "recorded_predictions",
        })
    return cases
