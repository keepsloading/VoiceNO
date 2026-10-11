"""Domain modules for VoiceNO Silent Speech Infrastructure.

VoiceNO is focused exclusively on the AI Prompts / Agent domain.
"""

from voiceno.domains.base import BaseDomain
from voiceno.domains.ai_prompts import get_ai_prompts_domain
from voiceno.domains.registry import DomainRegistry, get_active_domain, set_active_domain

__all__ = [
    "BaseDomain",
    "get_ai_prompts_domain",
    "DomainRegistry",
    "get_active_domain",
    "set_active_domain",
]
