"""
Prompt Dataset Loader
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PromptItem:
    """Single prompt item."""
    id: str
    prompt: str
    source: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "source": self.source,
            "metadata": self.metadata or {},
        }


class PromptDataset:
    """
    Load and manage prompt dataset.

    Supports:
    - JSONL format (one JSON object per line)
    - Sample size limiting
    - Source filtering
    """

    def __init__(
        self,
        data_path: str,
        sample_size: Optional[int] = None,
        source_filter: Optional[str] = None,
        shuffle: bool = False,
        seed: int = 42,
    ):
        """
        Args:
            data_path: Path to JSONL file
            sample_size: Limit number of samples (None = all)
            source_filter: Only include prompts from this source
            shuffle: Shuffle the dataset
            seed: Random seed for shuffling
        """
        self.data_path = data_path
        self.sample_size = sample_size
        self.source_filter = source_filter
        self.shuffle = shuffle
        self.seed = seed

        self.items: List[PromptItem] = []
        self._load()

    def _load(self):
        """Load prompts from file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        raw_items = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                prompt = obj.get("prompt", "").strip()
                if not prompt:
                    continue

                # Source filtering
                source = obj.get("source", "unknown")
                if self.source_filter and source != self.source_filter:
                    continue

                item = PromptItem(
                    id=str(obj.get("id", line_num)),
                    prompt=prompt,
                    source=source,
                    metadata={k: v for k, v in obj.items() if k not in ["id", "prompt", "source"]},
                )
                raw_items.append(item)

        # Shuffle if requested
        if self.shuffle:
            import random
            random.seed(self.seed)
            random.shuffle(raw_items)

        # Limit sample size
        if self.sample_size is not None:
            raw_items = raw_items[:self.sample_size]

        self.items = raw_items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> PromptItem:
        return self.items[idx]

    def __iter__(self):
        return iter(self.items)

    def get_prompts(self) -> List[str]:
        """Get all prompts as a list."""
        return [item.prompt for item in self.items]

    def summary(self) -> Dict[str, Any]:
        """Get dataset summary."""
        sources = {}
        for item in self.items:
            sources[item.source] = sources.get(item.source, 0) + 1

        return {
            "total": len(self.items),
            "data_path": self.data_path,
            "sample_size": self.sample_size,
            "source_distribution": sources,
        }