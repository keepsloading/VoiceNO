"""Contextual disambiguation and linguistic rescoring engine for VoiceNO.

Provides modular sentence and task context handling to rerank ambiguous
visual speech hypotheses without modifying the visual backbone.
"""

from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class TaskContext:
    """Environmental and task context cues for disambiguating silent speech."""
    active_app: Optional[str] = None          # e.g., 'browser', 'code_editor', 'document_viewer'
    selected_text: Optional[str] = None       # Text currently selected by the user
    preceding_text: Optional[str] = None      # Preceding words or conversational turn
    domain_keywords: List[str] = field(default_factory=list)  # Domain terms (e.g. ['python', 'error', 'water'])
    available_actions: List[str] = field(default_factory=list) # e.g. ['summarize', 'explain', 'search']


class ContextRescorer:
    """Reranks N-best visual speech hypotheses using linguistic priors and task context."""

    def __init__(
        self,
        alpha_linguistic: float = 0.40,
        beta_context: float = 0.60,
    ):
        self.alpha = alpha_linguistic
        self.beta = beta_context

        # Common English bigrams / collocations for visual disambiguation
        # Maps (w1, w2) -> log relative plausibility
        self.collocations = {
            ("GO", "TO"): 2.5,
            ("TO", "THE"): 3.0,
            ("THE", "MALL"): 2.2,
            ("THE", "HALL"): 1.8,
            ("THE", "BALL"): 1.4,
            ("DRINKING", "WATER"): 3.5,
            ("GET", "WATER"): 1.2,
            ("SUMMARIZE", "THIS"): 3.2,
            ("EXPLAIN", "THIS"): 3.0,
            ("REWRITE", "THIS"): 2.8,
            ("SEARCH", "FOR"): 3.0,
            ("OPEN", "MY"): 2.5,
            ("MY", "CALENDAR"): 3.0,
            ("TRANSLATE", "THIS"): 2.8,
            ("MAKE", "THIS"): 2.5,
            ("THIS", "SHORTER"): 2.6,
            ("DONT", "UNDERSTAND"): 3.2,
            ("WHY", "DOES"): 2.6,
            ("DOES", "THIS"): 2.5,
            ("THIS", "CODE"): 2.8,
            ("CODE", "FAIL"): 2.7,
            ("CHEAPEST", "OPTION"): 3.0,
            ("MORE", "PROFESSIONAL"): 2.9,
            ("WHAT", "THIS"): 2.4,
            ("THIS", "MEANS"): 2.8,
            ("PARAGRAPH", "CONTRADICTS"): 3.0,
            ("PREVIOUS", "SECTION"): 3.1,
            ("PACK", "MY"): 2.8,
            ("QUICK", "BROWN"): 3.5,
            ("BROWN", "FOX"): 3.5,
        }

        # App-action associations for task contextual boosting
        self.app_action_affinity = {
            "browser": ["SEARCH", "OPEN", "FIND", "GO", "PAGE", "CHEAPEST", "ARTICLE", "OPTION"],
            "code_editor": ["CODE", "FAIL", "EXPLAIN", "REWRITE", "ERROR", "FUNCTION", "DEBUG", "WHY"],
            "document_viewer": ["SUMMARIZE", "PARAGRAPH", "SECTION", "CONTRADICTS", "READ", "EXPLAIN", "REWRITE"],
            "calendar": ["OPEN", "MEETING", "CALENDAR", "TODAY", "SCHEDULE", "TOMORROW"],
            "general": ["WHAT", "WHY", "PLEASE", "MAKE", "SHORTER", "UNDERSTAND"],
        }

    def compute_linguistic_score(self, text: str) -> float:
        """Computes linguistic fluency score based on word transition priors."""
        words = re.sub(r"[^\w\s]", "", text.upper()).split()
        if not words:
            return -5.0

        score = 0.0
        # Penalize repeated consecutive words (e.g. "THEN THEY THEN THEY")
        repeats = 0
        for i in range(1, len(words)):
            if words[i] == words[i - 1]:
                repeats += 1
            pair = (words[i - 1], words[i])
            if pair in self.collocations:
                score += self.collocations[pair]
            else:
                score += 0.2  # baseline per-word positive transition

        # Heavily penalize repetitive looping artifacts common in unconstrained decoding
        score -= repeats * 4.0

        # Normalization by length to avoid pure length bias
        return score / max(1.0, len(words))

    def compute_context_score(self, text: str, context: Optional[TaskContext]) -> float:
        """Computes task relevance score based on active app, selected text, and keywords."""
        if context is None:
            return 0.0

        words = set(re.sub(r"[^\w\s]", "", text.upper()).split())
        if not words:
            return 0.0

        score = 0.0

        # 1. App affinity
        if context.active_app:
            app_key = context.active_app.lower()
            affinity_words = self.app_action_affinity.get(app_key, [])
            for w in words:
                if w in affinity_words:
                    score += 1.5

        # 2. Selected text & Domain keywords overlap
        target_context_tokens: Set[str] = set()
        if context.selected_text:
            target_context_tokens.update(re.sub(r"[^\w\s]", "", context.selected_text.upper()).split())
        if context.preceding_text:
            target_context_tokens.update(re.sub(r"[^\w\s]", "", context.preceding_text.upper()).split())
        for kw in context.domain_keywords:
            target_context_tokens.update(re.sub(r"[^\w\s]", "", kw.upper()).split())

        for w in words:
            if w in target_context_tokens:
                score += 2.0

        return score

    def rescore_hypotheses(
        self,
        hypotheses: List[Dict[str, Any]],
        context: Optional[TaskContext] = None,
    ) -> List[Dict[str, Any]]:
        """Reranks hypotheses by combining visual score, linguistic prior, and task context."""
        if not hypotheses:
            return hypotheses

        rescored_list: List[Dict[str, Any]] = []

        for h in hypotheses:
            text = h["text"]
            visual_score = h["score"]

            s_lang = self.compute_linguistic_score(text)
            s_ctx = self.compute_context_score(text, context)

            total_rescored = visual_score + (self.alpha * s_lang) + (self.beta * s_ctx)

            entry = dict(h)
            entry["visual_score"] = visual_score
            entry["linguistic_score"] = round(s_lang, 3)
            entry["context_score"] = round(s_ctx, 3)
            entry["rescored_score"] = round(total_rescored, 3)
            rescored_list.append(entry)

        # Sort by rescored_score descending
        rescored_list.sort(key=lambda x: x["rescored_score"], reverse=True)

        # Recompute normalized probabilities
        scores = [h["rescored_score"] for h in rescored_list]
        max_s = max(scores)
        exp_scores = [math.exp((s - max_s) / 2.0) for s in scores]
        sum_exp = sum(exp_scores) or 1.0

        for h, exp_s in zip(rescored_list, exp_scores):
            h["rescored_prob"] = round(exp_s / sum_exp, 4)

        return rescored_list
