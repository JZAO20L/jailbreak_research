"""
Base class for attack methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time


@dataclass
class AttackResult:
    """Result of an attack attempt."""

    # Core fields
    original_prompt: str
    attack_prompt: str
    strategy: str

    # Response fields
    target_response: str = ""
    guard_label: str = ""
    is_success: bool = False

    # Metadata
    iterations: int = 1
    time_cost: float = 0.0
    intermediate_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "attack_prompt": self.attack_prompt,
            "strategy": self.strategy,
            "target_response": self.target_response,
            "guard_label": self.guard_label,
            "is_success": self.is_success,
            "iterations": self.iterations,
            "time_cost": self.time_cost,
            "intermediate_results": self.intermediate_results,
            "metadata": self.metadata,
        }


class BaseAttacker(ABC):
    """Base class for all attack methods."""

    # Strategy name (override in subclass)
    name: str = "base"

    def __init__(
        self,
        target_client=None,
        guard_client=None,
        max_iterations: int = 10,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize attacker.

        Args:
            target_client: Client for target model (VLLMClient)
            guard_client: Client for guard model (VLLMClient)
            max_iterations: Maximum iterations for iterative methods
            verbose: Print verbose output
            **kwargs: Additional strategy-specific parameters
        """
        self.target_client = target_client
        self.guard_client = guard_client
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.kwargs = kwargs

    @abstractmethod
    def generate_attack_prompt(self, original_prompt: str) -> str:
        """
        Generate attack prompt from original prompt.

        Args:
            original_prompt: The original harmful prompt

        Returns:
            Generated attack prompt
        """
        pass

    def get_target_response(self, attack_prompt: str, **kwargs) -> str:
        """Get response from target model."""
        if self.target_client is None:
            raise ValueError("Target client not set")

        try:
            response = self.target_client.llm_call(
                prompt=attack_prompt,
                max_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.0),
            )
        except Exception as e:
            response = f"Error: {e}"

        return response

    def get_guard_label(self, attack_prompt: str, response: str) -> str:
        """Get label from guard model."""
        if self.guard_client is None:
            raise ValueError("Guard client not set")

        from src.prompts import GUARD_PROMPT

        messages = [
            {"role": "system", "content": GUARD_PROMPT},
            {"role": "user", "content": attack_prompt},
            {"role": "assistant", "content": response},
        ]

        try:
            raw = self.guard_client.llm_call(
                messages=messages,
                max_tokens=256,
                temperature=0.0,
            )
            from src.utils import extract_label_categories_refusal
            safe_label, categories, refusal = extract_label_categories_refusal(raw)

            if refusal and refusal.lower() == "yes":
                label = "refusal"
            elif safe_label and safe_label.lower() == "safe":
                label = "refusal"
            elif safe_label and safe_label.lower() == "unsafe":
                label = "success"
            else:
                label = "partial"
        except Exception as e:
            label = "unknown"

        return label

    def attack(
        self,
        original_prompt: str,
        evaluate: bool = True,
    ) -> AttackResult:
        """
        Execute attack on original prompt.

        Args:
            original_prompt: The original harmful prompt
            evaluate: Whether to evaluate the attack (get target response + guard label)

        Returns:
            AttackResult object
        """
        start_time = time.time()

        # Generate attack prompt
        attack_prompt, iterations, intermediate = self._generate_with_iterations(original_prompt)

        result = AttackResult(
            original_prompt=original_prompt,
            attack_prompt=attack_prompt,
            strategy=self.name,
            iterations=iterations,
            intermediate_results=intermediate,
        )

        # Evaluate if requested
        if evaluate and self.target_client is not None:
            result.target_response = self.get_target_response(attack_prompt)

            if self.guard_client is not None:
                result.guard_label = self.get_guard_label(attack_prompt, result.target_response)
                result.is_success = (result.guard_label == "success")

        result.time_cost = time.time() - start_time

        return result

    def _generate_with_iterations(self, original_prompt: str) -> tuple:
        """
        Generate attack prompt with iteration tracking.

        Returns:
            (attack_prompt, iterations, intermediate_results)
        """
        # Default: single iteration
        attack_prompt = self.generate_attack_prompt(original_prompt)
        return attack_prompt, 1, []

    def attack_batch(
        self,
        prompts: List[str],
        evaluate: bool = True,
        show_progress: bool = True,
    ) -> List[AttackResult]:
        """
        Attack multiple prompts.

        Args:
            prompts: List of original prompts
            evaluate: Whether to evaluate each attack
            show_progress: Show progress bar

        Returns:
            List of AttackResult objects
        """
        from tqdm import tqdm

        results = []
        iterator = tqdm(prompts, desc=f"[{self.name}]") if show_progress else prompts

        for prompt in iterator:
            result = self.attack(prompt, evaluate=evaluate)
            results.append(result)

        return results