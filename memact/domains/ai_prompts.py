"""AI Prompts Domain Module — The Beachhead Wedge for Memact.

Optimized for silent dictation and push-to-talk prompting of AI coding agents,
LLMs, Copilots, and developer terminal assistants.
"""

from typing import List, Tuple
from memact.domains.base import BaseDomain


def get_ai_prompts_domain() -> BaseDomain:
    """Returns the preconfigured AI Prompts domain."""
    action_verbs = [
        "CONVERT", "SUMMARIZE", "EXPLAIN", "RUN", "FIX", "DEBUG", "REFACTOR",
        "WRITE", "CREATE", "BUILD", "GENERATE", "OPTIMIZE", "OPEN", "SEARCH",
        "FIND", "FORMAT", "TEST", "TRANSLATE", "REWRITE", "ADD", "REMOVE",
    ]

    target_objects = [
        "THIS IMAGE", "THIS FILE", "THIS CODE", "THIS DOCUMENT", "THIS REPO",
        "THIS FUNCTION", "THIS CLASS", "THE TESTS", "THE BUG", "THE ERROR",
        "THE SCRIPT", "MY CALENDAR", "THIS PR", "THE DIFF", "THIS COMPONENT",
    ]

    domain_keywords = [
        "CONVERT", "IMAGE", "PDF", "SUMMARIZE", "DOCUMENT", "FILE", "CODE",
        "TESTS", "TEST", "BUG", "ERROR", "FIX", "DEBUG", "REFACTOR", "EXPLAIN",
        "SCRIPT", "REPO", "FUNCTION", "CLASS", "PROMPT", "CALENDAR", "BROWSER",
        "TERMINAL", "PYTHON", "TYPESCRIPT", "REACT", "DOCKER", "GIT", "API",
    ]

    grammar_patterns: List[Tuple[str, float]] = [
        # Action -> Target -> Format
        (r"\b(CONVERT|CHANGE|TRANSFORM)\b.*\b(IMAGE|FILE|DOCUMENT|CODE)\b.*\b(TO|INTO)\b.*\b(PDF|MARKDOWN|PNG|JSON)\b", 4.0),
        (r"\b(IMAGE)\b.*\b(TO)\b.*\b(PDF)\b", 3.5),
        # Agent Analysis / Refactor Commands
        (r"\b(SUMMARIZE|EXPLAIN|ANALYZE)\b.*\b(THIS|THE)\b.*\b(DOCUMENT|CODE|FILE|DIFF|REPO|TEXT)\b", 3.5),
        (r"\b(REFACTOR|OPTIMIZE|REWRITE)\b.*\b(THIS|THE)\b.*\b(FUNCTION|CLASS|CODE|COMPONENT)\b", 3.5),
        (r"\b(RUN|EXECUTE)\b.*\b(THE|ALL)\b.*\b(TESTS|TEST|SCRIPT|BUILD)\b", 3.5),
        (r"\b(FIX|DEBUG|RESOLVE)\b.*\b(THE|THIS)\b.*\b(BUG|ERROR|ISSUE|FAILING TEST)\b", 3.5),
        (r"\b(CREATE|BUILD|GENERATE|WRITE)\b.*\b(A|AN|NEW)\b.*\b(TEST|FUNCTION|SCRIPT|COMPONENT|API)\b", 3.0),
        (r"\b(OPEN|SHOW)\b.*\b(CALENDAR|BROWSER|TERMINAL|FILE|EDITOR)\b", 2.5),
        (r"\b(GIT)\b.*\b(COMMIT|PUSH|PULL|STATUS|DIFF)\b", 3.0),
    ]

    casual_penalty_fillers = [
        r"\bIN MY SCHOOL\b",
        r"\bLIKE IT'S\b",
        r"\bYOU KNOW WHAT\b",
        r"\bI WAS TALKING\b",
        r"\bSEE YOU IN FM\b",
        r"\bSAY TO ME FM\b",
        r"\bMAKING A FILM\b",
        r"\bI MAY SEE YOU\b",
    ]

    expansion_rules: List[Tuple[str, str]] = [
        # Bilabial co-articulation fixes before nouns
        (r"\b(WITH|IN|FOR|TO|OF|AT|ON|HAVE|BUY|IS|LIKE)\s+AM\b", r"\1 A"),
        (r"\b(WITH|IN|FOR|TO|OF|AT|ON|HAVE|BUY|IS|LIKE)\s+AM\b", r"\1 THE"),
        (r"\bAM\s+(BALL|CAR|DOG|CAT|BOOK|HOUSE|BIKE|PEN|KITE|FILE|IMAGE|PDF)\b", r"A \1"),
        # Viseme class swaps
        (r"\bBARK\b", "PARK"),
        (r"\bPARK\b", "BARK"),
        (r"\bBALL\b", "MALL"),
        (r"\bMALL\b", "BALL"),
        (r"\bBITE\b", "KITE"),
        (r"\bKITE\b", "BITE"),
    ]

    return BaseDomain(
        name="ai_prompts",
        description="Silent prompt and instruction dictation for AI agents, Copilots, and LLMs.",
        action_verbs=action_verbs,
        target_objects=target_objects,
        domain_keywords=domain_keywords,
        grammar_patterns=grammar_patterns,
        casual_penalty_fillers=casual_penalty_fillers,
        expansion_rules=expansion_rules,
    )
