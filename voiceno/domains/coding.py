"""Coding & Developer Syntax Domain Module for VoiceNO.

Optimized for silent dictation of programming constructs, variable names,
terminal commands, and git operations.
"""

from typing import List, Tuple
from voiceno.domains.base import BaseDomain


def get_coding_domain() -> BaseDomain:
    """Returns the preconfigured Coding & Syntax domain module."""
    action_verbs = [
        "DEFINE", "IMPORT", "RETURN", "CALL", "CREATE", "DECLARE", "ASSIGN",
        "COMMIT", "PUSH", "PULL", "CHECKOUT", "BRANCH", "MERGE", "CLONE",
        "RUN", "BUILD", "COMPILE", "DEBUG", "LOG", "PRINT", "RAISE", "CATCH",
    ]

    target_objects = [
        "FUNCTION", "VARIABLE", "CLASS", "METHOD", "INTERFACE", "TYPE",
        "ARRAY", "LIST", "DICT", "MAP", "STRING", "BOOLEAN", "PROMISE",
        "GIT STATUS", "GIT COMMIT", "GIT PUSH", "TERMINAL", "DOCKER CONTAINER",
    ]

    domain_keywords = [
        "FUNCTION", "VARIABLE", "CLASS", "INTERFACE", "TYPE", "IMPORT", "EXPORT",
        "RETURN", "CONST", "LET", "VAR", "DEF", "ASYNC", "AWAIT", "GIT", "COMMIT",
        "PUSH", "PULL", "BRANCH", "MERGE", "CHECKOUT", "STATUS", "DIFF", "LOG",
        "TERMINAL", "BASH", "DOCKER", "NPM", "PIP", "PYTEST", "TEST", "DEBUG",
        "PRINT", "CONSOLE", "EXCEPTION", "ERROR", "TRY", "CATCH", "IF", "ELSE",
        "FOR", "WHILE", "LOOP", "MODULE", "PACKAGE", "DEPENDENCY", "SCHEMA",
    ]

    grammar_patterns: List[Tuple[str, float]] = [
        # Git Commands
        (r"\bGIT\b.*\b(COMMIT|PUSH|PULL|STATUS|DIFF|CHECKOUT|BRANCH|MERGE|ADD|CLONE)\b", 4.0),
        # Code Declarations
        (r"\b(DEF|FUNCTION|CONST|LET|VAR|CLASS)\b.*\b\w+\b", 3.5),
        (r"\b(IMPORT|FROM|REQUIRE)\b.*\b\w+\b", 3.5),
        (r"\b(RETURN|YIELD)\b.*\b\w+\b", 3.0),
        (r"\b(TRY|CATCH|EXCEPT|FINALLY|RAISE|THROW)\b", 3.0),
        (r"\b(ASYNC|AWAIT|PROMISE)\b", 3.0),
        (r"\b(PRINT|CONSOLE\.LOG|LOGGER)\b", 3.0),
        (r"\b(NPM|PIP|DOCKER|CARGO|POETRY)\b.*\b(RUN|INSTALL|BUILD|TEST)\b", 3.5),
    ]

    casual_penalty_fillers = [
        r"\bIN MY SCHOOL\b",
        r"\bLIKE IT'S\b",
        r"\bYOU KNOW WHAT\b",
        r"\bSEE YOU IN FM\b",
        r"\bMAKING A FILM\b",
    ]

    expansion_rules: List[Tuple[str, str]] = [
        (r"\bGIT COMET\b", "GIT COMMIT"),
        (r"\bGIT PUSS\b", "GIT PUSH"),
        (r"\bGET STATUS\b", "GIT STATUS"),
        (r"\bGET PULL\b", "GIT PULL"),
        (r"\bGET COMMIT\b", "GIT COMMIT"),
        (r"\bCON'S\b", "CONST"),
        (r"\bLET'S\b", "LET"),
    ]

    return BaseDomain(
        name="coding",
        description="Silent dictation of code, developer syntax, terminal commands, and git operations.",
        action_verbs=action_verbs,
        target_objects=target_objects,
        domain_keywords=domain_keywords,
        grammar_patterns=grammar_patterns,
        casual_penalty_fillers=casual_penalty_fillers,
        expansion_rules=expansion_rules,
    )
