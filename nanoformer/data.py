"""Tiny data pipeline: tokenize a text corpus into a flat tensor of ids and
serve random contiguous windows for next-token prediction.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .tokenizer import BPETokenizer


class TokenDataset:
    """Holds a 1-D tensor of token ids and yields random (x, y) windows."""

    def __init__(self, ids: torch.Tensor, block_size: int, val_fraction: float = 0.1):
        n_val = int(len(ids) * val_fraction)
        self.block_size = block_size
        self.splits = {"train": ids[:-n_val], "val": ids[-n_val:]}

    def get_batch(self, split: str, batch_size: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        data = self.splits[split]
        max_start = len(data) - self.block_size - 1
        idx = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in idx])
        y = torch.stack([data[i + 1 : i + 1 + self.block_size] for i in idx])
        return x.to(device), y.to(device)

    @classmethod
    def from_text(
        cls, text: str, tokenizer: BPETokenizer, block_size: int, val_fraction: float = 0.1
    ) -> "TokenDataset":
        ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        return cls(ids, block_size, val_fraction)

    @classmethod
    def from_file(
        cls, path: str | Path, tokenizer: BPETokenizer, block_size: int, val_fraction: float = 0.1
    ) -> "TokenDataset":
        return cls.from_text(Path(path).read_text(), tokenizer, block_size, val_fraction)
