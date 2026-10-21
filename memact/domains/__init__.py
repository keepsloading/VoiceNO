"""Domain modules for Memact Silent Speech Infrastructure.

Memact is focused exclusively on the AI Prompts / Agent domain.
"""

from memact.domains.base import BaseDomain
from memact.domains.ai_prompts import get_ai_prompts_domain
from memact.domains.registry import DomainRegistry, get_active_domain, set_active_domain

__all__ = [
    "BaseDomain",
    "get_ai_prompts_domain",
    "DomainRegistry",
    "get_active_domain",
    "set_active_domain",
]
