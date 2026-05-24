"""
Baseline Attack Methods for Jailbreak Prompt Synthesis

This module provides various attack strategies for generating jailbreak prompts.

Note: no_rewrite and template_rewrite have been removed.
"""

from .base import BaseAttacker, AttackResult
from .multilingual import MultilingualAttacker, MultilingualEnsembleAttacker
from .pair import PAIRAttacker
from .genetic import GeneticAttacker
from .deepinception import DeepInceptionAttacker, DeepInceptionMultiLayerAttacker, PersonaModulationAttacker

__all__ = [
    "BaseAttacker",
    "AttackResult",
    "MultilingualAttacker",
    "MultilingualEnsembleAttacker",
    "PAIRAttacker",
    "GeneticAttacker",
    "DeepInceptionAttacker",
    "DeepInceptionMultiLayerAttacker",
    "PersonaModulationAttacker",
]

# Strategy registry
STRATEGIES = {
    "multilingual": MultilingualAttacker,
    "multilingual_ensemble": MultilingualEnsembleAttacker,
    "pair": PAIRAttacker,
    "genetic": GeneticAttacker,
    "deepinception": DeepInceptionAttacker,
    "deepinception_multilayer": DeepInceptionMultiLayerAttacker,
    "persona": PersonaModulationAttacker,
}


def get_attacker(name: str, **kwargs):
    """Get attacker by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name](**kwargs)


def list_strategies():
    """List all available strategies."""
    return list(STRATEGIES.keys())