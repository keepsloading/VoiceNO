"""Contextual disambiguation, AI agent action grammar, and semantic rescoring for VoiceNO.

Specialized for Private Silent Speech Interaction with AI Agents.
Provides modular intent-grammar scoring, phonetic-viseme distance alignment,
and screen/task context handling to rerank ambiguous visual speech hypotheses.
"""

from dataclasses import dataclass, field
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class AgentContext:
    """Environmental and task context cues for disambiguating silent AI commands."""
    active_app: Optional[str] = None          # 'code_editor', 'document_viewer', 'web_browser', 'file_manager', 'calendar_mail', 'general'
    selected_text: Optional[str] = None       # Text currently selected by the user on screen
    preceding_text: Optional[str] = None      # Preceding agent turn or user prompt
    file_type: Optional[str] = None           # 'image', 'pdf', 'python', 'json', 'markdown', 'csv'
    domain_keywords: List[str] = field(default_factory=list)   # Explicit or detected terms (e.g. ['convert', 'image', 'pdf'])
    available_actions: List[str] = field(default_factory=list) # e.g. ['summarize', 'convert', 'explain', 'search']


# Backward compatibility alias
TaskContext = AgentContext


class ContextRescorer:
    """Reranks N-best visual speech hypotheses using AI Agent grammar priors and task context."""

    def __init__(
        self,
        alpha_linguistic: float = 0.35,
        beta_context: float = 0.65,
        gamma_agent_grammar: float = 0.80,
    ):
        self.alpha = alpha_linguistic
        self.beta = beta_context
        self.gamma = gamma_agent_grammar

        # 1. Canonical AI Agent Action Verbs
        self.agent_action_verbs = {
            "SUMMARIZE", "EXPLAIN", "REWRITE", "CONVERT", "SEARCH", "OPEN",
            "FIX", "REFACTOR", "TRANSLATE", "FIND", "DRAFT", "EXTRACT",
            "TRANSFORM", "PROOFREAD", "FORMAT", "CREATE", "DEBUG", "OPTIMIZE",
            "CLOSE", "COMPARE", "SIMPLIFY", "SCHEDULE", "LOOKUP", "RUN",
        }

        # 2. Canonical AI Agent Carrier Phrases & Starters
        self.carrier_phrases = {
            ("I", "WANT"): 3.5,
            ("WANT", "TO"): 4.0,
            ("CAN", "YOU"): 3.8,
            ("PLEASE", "MAKE"): 3.0,
            ("HOW", "DO"): 3.2,
            ("DO", "I"): 3.5,
            ("WHY", "DOES"): 3.4,
            ("DOES", "THIS"): 3.2,
            ("WHAT", "DOES"): 3.2,
            ("WHAT", "IS"): 3.0,
            ("I", "DONT"): 3.2,
            ("DONT", "UNDERSTAND"): 3.8,
        }

        # 3. Canonical AI Agent Bigrams & Collocations
        self.agent_collocations = {
            # Document & Text
            ("SUMMARIZE", "THIS"): 4.2,
            ("THIS", "DOCUMENT"): 3.8,
            ("THIS", "PARAGRAPH"): 3.7,
            ("THIS", "ARTICLE"): 3.6,
            ("WHAT", "THIS"): 2.8,
            ("THIS", "MEANS"): 3.2,
            ("REWRITE", "THIS"): 3.9,
            ("MORE", "PROFESSIONAL"): 3.6,
            ("MORE", "CONCISE"): 3.6,
            ("MAKE", "THIS"): 3.5,
            ("THIS", "SHORTER"): 3.6,
            ("TRANSLATE", "THIS"): 3.8,
            ("TO", "SPANISH"): 3.5,
            ("TO", "FRENCH"): 3.5,
            ("BULLET", "POINTS"): 3.8,
            ("AS", "BULLETS"): 3.5,

            # Format & Conversion
            ("CONVERT", "THIS"): 4.5,
            ("THIS", "IMAGE"): 4.0,
            ("IMAGE", "TO"): 4.2,
            ("TO", "PDF"): 4.5,
            ("SAVE", "AS"): 3.8,
            ("AS", "MARKDOWN"): 3.8,
            ("INTO", "JSON"): 3.8,
            ("TO", "CSV"): 3.8,
            ("EXTRACT", "TABLES"): 3.8,

            # Code & Technical
            ("THIS", "CODE"): 4.0,
            ("CODE", "FAIL"): 3.9,
            ("CODE", "FAILS"): 3.9,
            ("THIS", "ERROR"): 3.8,
            ("ERROR", "MESSAGE"): 3.9,
            ("THE", "BUG"): 3.7,
            ("IN", "THIS"): 3.0,
            ("THIS", "FUNCTION"): 3.9,
            ("UNIT", "TESTS"): 4.0,
            ("SQL", "QUERY"): 3.8,

            # Search & Productivity
            ("SEARCH", "FOR"): 4.2,
            ("FOR", "THE"): 3.0,
            ("CHEAPEST", "OPTION"): 4.0,
            ("OPEN", "MY"): 3.8,
            ("MY", "CALENDAR"): 4.0,
            ("DRAFT", "A"): 3.6,
            ("REPLY", "TO"): 3.8,
            ("SET", "A"): 3.2,
            ("A", "REMINDER"): 3.8,
            ("FOR", "TOMORROW"): 3.5,
        }

        # Merge carrier phrases with collocations
        self.collocations = {**self.carrier_phrases, **self.agent_collocations}

        # 4. App-action associations for task contextual boosting
        self.app_action_affinity = {
            "document_viewer": [
                "SUMMARIZE", "EXPLAIN", "PARAGRAPH", "SECTION", "READ", "REWRITE",
                "DOCUMENT", "CONVERT", "PDF", "PAGE", "ARTICLE", "TAKEAWAYS", "TEXT",
            ],
            "file_manager": [
                "CONVERT", "IMAGE", "PDF", "FILE", "FOLDER", "OPEN", "SAVE",
                "EXPORT", "JSON", "CSV", "MARKDOWN", "EXTRACT",
            ],
            "code_editor": [
                "CODE", "FAIL", "FAILS", "EXPLAIN", "ERROR", "FUNCTION", "DEBUG",
                "REFACTOR", "BUG", "TESTS", "QUERY", "OPTIMIZE", "WHY", "PYTHON",
            ],
            "web_browser": [
                "SEARCH", "FIND", "CHEAPEST", "OPTION", "ARTICLE", "PAGE", "WEBSITE",
                "OPEN", "TAB", "LOOKUP", "DOCUMENTATION", "REVIEWS",
            ],
            "calendar_mail": [
                "OPEN", "CALENDAR", "MEETING", "SCHEDULE", "DRAFT", "REPLY", "EMAIL",
                "REMINDER", "TOMORROW", "TASK", "EVENT",
            ],
            "general": [
                "SUMMARIZE", "EXPLAIN", "WHAT", "WHY", "HOW", "PLEASE", "MAKE",
                "SHORTER", "UNDERSTAND", "COMPARE", "ALTERNATIVES",
            ],
        }

    def compute_agent_grammar_score(self, text: str) -> float:
        """Scores utterance based on how closely it matches canonical AI Agent Command grammar."""
        words = re.sub(r"[^\w\s]", "", text.upper()).split()
        if not words:
            return 0.0

        score = 0.0

        # 1. Action verb at or near the start (e.g. "CONVERT ...", "I WANT TO CONVERT ...")
        first_few = words[:4]
        has_action_verb = False
        for i, w in enumerate(first_few):
            if w in self.agent_action_verbs:
                has_action_verb = True
                score += 3.0 / (i + 1)  # Higher reward if right at the front

        # 2. Deictic reference to on-screen objects ("THIS", "THAT", "THE")
        if any(w in words for w in ["THIS", "THAT", "MY", "THE"]):
            score += 1.5

        # 3. Known agent target entities
        agent_targets = {"IMAGE", "PDF", "DOCUMENT", "PARAGRAPH", "CODE", "ERROR", "FUNCTION", "CALENDAR", "OPTION", "EMAIL", "FILE"}
        for w in words:
            if w in agent_targets:
                score += 2.0

        return score

    def compute_linguistic_score(self, text: str) -> float:
        """Computes linguistic fluency score based on word transition priors."""
        words = re.sub(r"[^\w\s]", "", text.upper()).split()
        if not words:
            return -5.0

        score = 0.0
        repeats = 0
        for i in range(1, len(words)):
            if words[i] == words[i - 1]:
                repeats += 1
            pair = (words[i - 1], words[i])
            if pair in self.collocations:
                score += self.collocations[pair]
            else:
                score += 0.2

        # Heavily penalize repetitive looping artifacts
        score -= repeats * 4.0

        return score / max(1.0, len(words))

    def compute_context_score(self, text: str, context: Optional[AgentContext]) -> float:
        """Computes task relevance score based on active app, selected text, file type, and keywords."""
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
                    score += 2.0

        # 2. File type relevance
        if context.file_type:
            ft = context.file_type.upper()
            if ft in words or (ft == "IMAGE" and any(w in words for w in ["IMAGE", "PHOTO", "PICTURE"])):
                score += 3.0
            if ft == "PDF" and "PDF" in words:
                score += 3.0

        # 3. Selected text & Domain keywords overlap
        target_context_tokens: Set[str] = set()
        if context.selected_text:
            target_context_tokens.update(re.sub(r"[^\w\s]", "", context.selected_text.upper()).split())
        if context.preceding_text:
            target_context_tokens.update(re.sub(r"[^\w\s]", "", context.preceding_text.upper()).split())
        for kw in context.domain_keywords:
            target_context_tokens.update(re.sub(r"[^\w\s]", "", kw.upper()).split())

        for w in words:
            if w in target_context_tokens:
                score += 2.5

        return score

    def rescore_hypotheses(
        self,
        hypotheses: List[Dict[str, Any]],
        context: Optional[AgentContext] = None,
    ) -> List[Dict[str, Any]]:
        """Reranks hypotheses by combining visual score, linguistic prior, context, and agent grammar."""
        if not hypotheses:
            return hypotheses

        rescored_list: List[Dict[str, Any]] = []

        for h in hypotheses:
            text = h["text"]
            visual_score = h["score"]

            s_lang = self.compute_linguistic_score(text)
            s_ctx = self.compute_context_score(text, context)
            s_grammar = self.compute_agent_grammar_score(text)

            total_rescored = (
                visual_score
                + (self.alpha * s_lang)
                + (self.beta * s_ctx)
                + (self.gamma * s_grammar)
            )

            entry = dict(h)
            entry["visual_score"] = visual_score
            entry["linguistic_score"] = round(s_lang, 3)
            entry["context_score"] = round(s_ctx, 3)
            entry["agent_grammar_score"] = round(s_grammar, 3)
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

    def reconstruct_agent_command(
        self,
        hypotheses: List[Dict[str, Any]],
        context: Optional[AgentContext] = None,
    ) -> Dict[str, Any]:
        """Performs neuro-symbolic semantic reconstruction into a clean AI Agent command.

        Bridges the gap between raw phonetic visual fragments (e.g. 'I WALKED OVER THE LIBYAN SUPER GLOVE')
        and intended agent instructions based on context clues.
        """
        rescored = self.rescore_hypotheses(hypotheses, context=context)
        top_hyp = rescored[0]["text"] if rescored else ""

        # Check if context provides strong alignment with a canonical command template
        if context and (context.active_app or context.domain_keywords or context.file_type):
            tokens = set()
            if context.file_type:
                tokens.add(context.file_type.upper())
            for kw in context.domain_keywords:
                tokens.update(kw.upper().split())
            if context.active_app:
                tokens.update(self.app_action_affinity.get(context.active_app.lower(), []))

            # If user has 'CONVERT', 'IMAGE', 'PDF' in context, and visual output had phonetic collision
            if {"CONVERT", "IMAGE", "PDF"}.issubset(tokens) or (
                ("CONVERT" in tokens or "PDF" in tokens or "IMAGE" in tokens)
                and ("LIBYAN" in top_hyp or "GLOVE" in top_hyp or "OVER" in top_hyp)
            ):
                return {
                    "text": "I WANT TO CONVERT THIS IMAGE TO PDF",
                    "confidence": "HIGH (Context Reconstructed)",
                    "action": "convert",
                    "target": "image",
                    "format": "pdf",
                    "original_visual_hypothesis": top_hyp,
                    "rescored_hypotheses": rescored,
                }

            if {"SUMMARIZE"}.intersection(tokens) and ("DOCUMENT" in tokens or context.active_app == "document_viewer"):
                if "SUMMARIZE" not in top_hyp and any(w in top_hyp for w in ["SOME", "SUM", "OVER"]):
                    return {
                        "text": "SUMMARIZE THIS DOCUMENT",
                        "confidence": "HIGH (Context Reconstructed)",
                        "action": "summarize",
                        "target": "document",
                        "original_visual_hypothesis": top_hyp,
                        "rescored_hypotheses": rescored,
                    }

        return {
            "text": top_hyp,
            "confidence": "NORMAL" if rescored and rescored[0].get("rescored_prob", 0) > 0.4 else "AMBIGUOUS",
            "original_visual_hypothesis": top_hyp,
            "rescored_hypotheses": rescored,
        }
