"""
Baseline Attack Methods for Jailbreak Prompt Synthesis

This module provides various attack strategies for generating jailbreak prompts.
"""

from .base import BaseAttacker, AttackResult
from .no_rewrite import NoRewriteAttacker
from .multilingual import MultilingualAttacker, MultilingualEnsembleAttacker
from .pair import PAIRAttacker
from .genetic import GeneticAttacker
from .deepinception import DeepInceptionAttacker, DeepInceptionMultiLayerAttacker, PersonaModulationAttacker
from .autodan import AutoDANAttacker
from .crescendo import CrescendoAttacker
from .tap import TAPAttacker

__all__ = [
    "BaseAttacker",
    "AttackResult",
    "NoRewriteAttacker",
    "MultilingualAttacker",
    "MultilingualEnsembleAttacker",
    "PAIRAttacker",
    "GeneticAttacker",
    "DeepInceptionAttacker",
    "DeepInceptionMultiLayerAttacker",
    "PersonaModulationAttacker",
    "AutoDANAttacker",
    "CrescendoAttacker",
    "TAPAttacker",
]

# Strategy registry
STRATEGIES = {
    "no_rewrite": NoRewriteAttacker,
    "multilingual": MultilingualAttacker,
    "multilingual_ensemble": MultilingualEnsembleAttacker,
    "pair": PAIRAttacker,
    "genetic": GeneticAttacker,
    "deepinception": DeepInceptionAttacker,
    "deepinception_multilayer": DeepInceptionMultiLayerAttacker,
    "persona": PersonaModulationAttacker,
    "autodan": AutoDANAttacker,
    "crescendo": CrescendoAttacker,
    "tap": TAPAttacker,
}


def get_attacker(name: str, **kwargs):
    """Get attacker by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name](**kwargs)


def list_strategies():
    """List all available strategies."""
    return list(STRATEGIES.keys())