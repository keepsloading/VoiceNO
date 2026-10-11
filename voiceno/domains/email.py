"""Email & Workplace Communication Domain Module for VoiceNO.

Optimized for silent dictation of professional correspondence, Slack/Teams
messages, meeting scheduling, and project updates.
"""

from typing import List, Tuple
from voiceno.domains.base import BaseDomain


def get_email_domain() -> BaseDomain:
    """Returns the preconfigured Email & Workplace Communication domain module."""
    action_verbs = [
        "SEND", "SCHEDULE", "ATTACH", "CONFIRM", "FOLLOW", "UPDATE", "REVIEW",
        "APPROVE", "FORWARD", "REPLY", "CONNECT", "DISCUSS", "SHARE", "MEET",
    ]

    target_objects = [
        "EMAIL", "MEETING", "CALENDAR", "INVITE", "ATTACHMENT", "PROJECT",
        "TIMELINE", "DEADLINE", "UPDATE", "AGENDA", "SLIDES", "DECK", "NOTES",
    ]

    domain_keywords = [
        "EMAIL", "MEETING", "CALENDAR", "SCHEDULE", "ATTACHED", "ATTACHMENT",
        "UPDATE", "PROJECT", "TIMELINE", "DEADLINE", "REVIEW", "APPROVE",
        "CONFIRM", "REGARDS", "BEST", "THANKS", "THANK", "PLEASE", "FOLLOW",
        "SLACK", "TEAMS", "AGENDA", "NOTES", "DISCUSSION", "SYNC", "CALL",
    ]

    grammar_patterns: List[Tuple[str, float]] = [
        # Polite Correspondence Openers & Closers
        (r"\b(PLEASE|COULD YOU|CAN YOU)\b.*\b(LET ME KNOW|REVIEW|CONFIRM|SEND|SHARE)\b", 3.5),
        (r"\b(THANKS|THANK YOU)\b.*\b(FOR THE UPDATE|FOR YOUR TIME|FOR JOINING|IN ADVANCE)\b", 3.5),
        (r"\b(LOOK FORWARD TO)\b.*\b(HEARING|MEETING|SEEING|WORKING)\b", 3.5),
        (r"\b(LET'S|LET US)\b.*\b(SYNC|SCHEDULE|CONNECT|MEET)\b", 3.0),
        (r"\b(ATTACHED|ATTACHING)\b.*\b(IS THE|ARE THE|DOCUMENT|DECK|SLIDES|FILE)\b", 3.5),
        (r"\b(FOLLOWING UP|QUICK UPDATE|JUST CHECKING IN)\b", 3.5),
    ]

    casual_penalty_fillers = [
        r"\bIN MY SCHOOL\b",
        r"\bSEE YOU IN FM\b",
        r"\bMAKING A FILM\b",
    ]

    expansion_rules: List[Tuple[str, str]] = [
        (r"\bSINK UP\b", "SYNC UP"),
        (r"\bMEAT UP\b", "MEET UP"),
        (r"\bTHANKS FOR THE DATE\b", "THANKS FOR THE UPDATE"),
        (r"\bATTACH IT IS\b", "ATTACHED IS"),
    ]

    return BaseDomain(
        name="email",
        description="Silent dictation of professional workplace communication, emails, and calendar invites.",
        action_verbs=action_verbs,
        target_objects=target_objects,
        domain_keywords=domain_keywords,
        grammar_patterns=grammar_patterns,
        casual_penalty_fillers=casual_penalty_fillers,
        expansion_rules=expansion_rules,
    )
