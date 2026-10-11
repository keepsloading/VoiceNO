"""Domain modules for VoiceNO Silent Speech Infrastructure."""

from voiceno.domains.base import BaseDomain
from voiceno.domains.ai_prompts import get_ai_prompts_domain
from voiceno.domains.coding import get_coding_domain
from voiceno.domains.email import get_email_domain
from voiceno.domains.blogging import get_blogging_domain
from voiceno.domains.registry import DomainRegistry, get_active_domain, set_active_domain

__all__ = [
    "BaseDomain",
    "get_ai_prompts_domain",
    "get_coding_domain",
    "get_email_domain",
    "get_blogging_domain",
    "DomainRegistry",
    "get_active_domain",
    "set_active_domain",
]

