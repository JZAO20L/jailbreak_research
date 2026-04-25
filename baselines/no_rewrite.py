"""
No-Rewrite Attack: Use original prompt directly.
"""

from .base import BaseAttacker


class NoRewriteAttacker(BaseAttacker):
    """Direct attack without any modification to the original prompt."""

    name = "no_rewrite"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_attack_prompt(self, original_prompt: str) -> str:
        """Return the original prompt unchanged."""
        return original_prompt