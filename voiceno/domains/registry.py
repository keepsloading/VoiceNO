"""Registry and manager for VoiceNO Domain Modules."""

from typing import Dict, List, Optional
from voiceno.domains.base import BaseDomain
from voiceno.domains.ai_prompts import get_ai_prompts_domain


class DomainRegistry:
    """Manages active and registered silent speech domain modules."""

    _domains: Dict[str, BaseDomain] = {}
    _active_domain: str = "ai_prompts"

    @classmethod
    def register(cls, domain: BaseDomain) -> None:
        """Registers a domain module."""
        cls._domains[domain.name] = domain

    @classmethod
    def get(cls, name: Optional[str] = None) -> BaseDomain:
        """Returns the requested domain or the currently active domain."""
        cls._ensure_defaults()
        domain_name = name or cls._active_domain
        if domain_name not in cls._domains:
            raise KeyError(f"Domain '{domain_name}' not found. Available: {list(cls._domains.keys())}")
        return cls._domains[domain_name]

    @classmethod
    def set_active(cls, name: str) -> BaseDomain:
        """Sets the global active domain module."""
        cls._ensure_defaults()
        if name not in cls._domains:
            raise KeyError(f"Cannot activate unknown domain '{name}'. Available: {list(cls._domains.keys())}")
        cls._active_domain = name
        return cls._domains[name]

    @classmethod
    def list_domains(cls) -> List[str]:
        """Lists all registered domain names."""
        cls._ensure_defaults()
        return list(cls._domains.keys())

    @classmethod
    def _ensure_defaults(cls) -> None:
        """Initializes default built-in domain modules."""
        if "ai_prompts" not in cls._domains:
            cls.register(get_ai_prompts_domain())


# Convenience functions
def get_active_domain() -> BaseDomain:
    return DomainRegistry.get()


def set_active_domain(name: str) -> BaseDomain:
    return DomainRegistry.set_active(name)
