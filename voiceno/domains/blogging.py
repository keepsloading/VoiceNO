"""Blogging & Long-Form Writing Domain Module for VoiceNO.

Optimized for silent dictation of essays, technical blogs, articles,
thought-to-text drafting, and creative prose.
"""

from typing import List, Tuple
from voiceno.domains.base import BaseDomain


def get_blogging_domain() -> BaseDomain:
    """Returns the preconfigured Blogging & Long-Form Writing domain module."""
    action_verbs = [
        "WRITE", "DRAFT", "EXPLAIN", "ILLUSTRATE", "EXPLORE", "ARGUE", "PROPOSE",
        "CONCLUDE", "INTRODUCE", "HIGHLIGHT", "ANALYZE", "DESCRIBE", "REFLECT",
    ]

    target_objects = [
        "ARTICLE", "BLOG POST", "ESSAY", "PARAGRAPH", "INTRODUCTION",
        "CONCLUSION", "PERSPECTIVE", "ARGUMENT", "INSIGHT", "STORY", "SECTION",
    ]

    domain_keywords = [
        "ARTICLE", "BLOG", "POST", "ESSAY", "PARAGRAPH", "INTRODUCTION",
        "CONCLUSION", "PERSPECTIVE", "ARGUMENT", "INSIGHT", "EXAMPLE",
        "STORY", "CHAPTER", "SECTION", "AUDIENCE", "DRAFT", "HEADLINE",
        "SUBTITLE", "HOWEVER", "THEREFORE", "FURTHERMORE", "MOREOVER",
    ]

    grammar_patterns: List[Tuple[str, float]] = [
        (r"\b(IN THIS ARTICLE|IN THIS POST|IN THIS ESSAY)\b", 3.5),
        (r"\b(FIRST OF ALL|FIRST AND FOREMOST|TO BEGIN WITH)\b", 3.0),
        (r"\b(ON THE OTHER HAND|AT THE SAME TIME|IN CONTRAST)\b", 3.0),
        (r"\b(AS A RESULT|CONSEQUENTLY|THEREFORE|HOWEVER)\b", 3.0),
        (r"\b(IN CONCLUSION|TO SUM UP|ULTIMATELY)\b", 3.5),
        (r"\b(THE KEY TAKEAWAY|AN IMPORTANT POINT)\b", 3.5),
    ]

    casual_penalty_fillers = [
        r"\bSEE YOU IN FM\b",
        r"\bMAKING A FILM\b",
    ]

    expansion_rules: List[Tuple[str, str]] = [
        (r"\bIN THIS R TICKLE\b", "IN THIS ARTICLE"),
        (r"\bFIRST OFF ALL\b", "FIRST OF ALL"),
    ]

    return BaseDomain(
        name="blogging",
        description="Silent dictation of articles, technical blog posts, essays, and creative writing.",
        action_verbs=action_verbs,
        target_objects=target_objects,
        domain_keywords=domain_keywords,
        grammar_patterns=grammar_patterns,
        casual_penalty_fillers=casual_penalty_fillers,
        expansion_rules=expansion_rules,
    )
