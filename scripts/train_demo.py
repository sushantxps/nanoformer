"""End-to-end demo: train a tiny model on a text corpus and sample from it.

Usage:
    python scripts/train_demo.py --input data.txt --steps 2000

If no input file is given it falls back to a small built-in corpus so the
repo runs out of the box with zero downloads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nanoformer import (
    BPETokenizer,
    ModelConfig,
    Transformer,
    TokenDataset,
    TrainConfig,
    Trainer,
    generate,
)

FALLBACK_CORPUS = (
    "All happy families are alike; each unhappy family is unhappy in its own way. "
    "It was the best of times, it was the worst of times. "
    "Call me Ishmael. Some years ago, never mind how long precisely. "
    "In the beginning the universe was created. This has made a lot of people very angry. "
) * 400


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=None)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--out", type=str, default="checkpoint.pt")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    text = Path(args.input).read_text() if args.input else FALLBACK_CORPUS

    print("training tokenizer...")
    tok = BPETokenizer()
    tok.train(text, vocab_size=args.vocab)
    tok.save("tokenizer.json")

    dataset = TokenDataset.from_text(text, tok, block_size=args.block)
    print(f"train tokens: {len(dataset.splits['train'])}, "
          f"val tokens: {len(dataset.splits['val'])}")

    model_cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        dim=args.dim,
        n_layers=args.layers,
        n_heads=6,
        n_kv_heads=2,
        max_seq_len=args.block,
    )
    model = Transformer(model_cfg)
    print(f"model parameters: {model.num_parameters():,}")

    train_cfg = TrainConfig(
        lr=3e-4,
        max_steps=args.steps,
        warmup_steps=max(20, args.steps // 20),
        eval_interval=max(100, args.steps // 5),
    )

    def get_batch(split: str):
        return dataset.get_batch(split, args.batch, device)

    trainer = Trainer(model, train_cfg, get_batch, device=device)
    history = trainer.train()
    trainer.save_checkpoint(args.out)
    Path("history.json").write_text(json.dumps(history))

    print("\n--- sample ---")
    prompt = torch.tensor([tok.encode("It was", bos=False)], dtype=torch.long)
    out = generate(model, prompt, max_new_tokens=60, temperature=0.8, top_k=40)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
