"""Base domain definition for VoiceNO Domain-Modular Silent Speech Infrastructure."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BaseDomain:
    """Represents a specialized domain module with tailored vocabulary, intent grammars, and priors."""

    name: str
    description: str
    action_verbs: List[str] = field(default_factory=list)
    target_objects: List[str] = field(default_factory=list)
    domain_keywords: List[str] = field(default_factory=list)
    grammar_patterns: List[Tuple[str, float]] = field(default_factory=list)
    casual_penalty_fillers: List[str] = field(default_factory=list)
    expansion_rules: List[Tuple[str, str]] = field(default_factory=list)

    def compute_domain_bias(self, text: str) -> float:
        """Computes domain-specific log-likelihood bias for candidate text."""
        import re

        clean = text.upper().strip()
        tokens = set(re.findall(r"\b\w+\b", clean))
        bias = 0.0

        # Match domain keywords
        if self.domain_keywords:
            matched = tokens.intersection({kw.upper() for kw in self.domain_keywords})
            bias += 1.8 * len(matched)

        # Match structured domain grammar patterns
        for pattern, boost in self.grammar_patterns:
            if re.search(pattern, clean):
                bias += boost
                break

        # Penalize out-of-domain conversational noise
        for filler in self.casual_penalty_fillers:
            if re.search(filler, clean):
                bias -= 3.5

        return bias
