"""nanoformer: a small, from-scratch LLM with a modern architecture."""

from .model import ModelConfig, Transformer
from .tokenizer import BPETokenizer
from .data import TokenDataset
from .train import TrainConfig, Trainer, cosine_lr, configure_optimizer
from .generate import generate, speculative_generate

__version__ = "0.1.0"

__all__ = [
    "ModelConfig",
    "Transformer",
    "BPETokenizer",
    "TokenDataset",
    "TrainConfig",
    "Trainer",
    "cosine_lr",
    "configure_optimizer",
    "generate",
    "speculative_generate",
]
