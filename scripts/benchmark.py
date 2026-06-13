"""Micro-benchmarks for the inference engine.

Measures two things on whatever device is available:
  1. KV-cache speedup: cached decoding vs. the naive O(n^2) recompute-each-step.
  2. Speculative decoding: tokens accepted per round and wall-clock latency,
     using a small draft model against a larger target.

Numbers are illustrative on CPU; the relative trends are what matter.
"""

from __future__ import annotations

import time

import torch

from nanoformer import ModelConfig, Transformer, generate, speculative_generate


def time_it(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


@torch.no_grad()
def naive_generate(model: Transformer, prompt: torch.Tensor, n: int) -> torch.Tensor:
    """No cache: re-encode the full sequence every step (the baseline)."""
    tokens = prompt
    for _ in range(n):
        logits, _, _ = model(tokens)
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        tokens = torch.cat([tokens, nxt], dim=1)
    return tokens


def main() -> None:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\n")

    cfg = ModelConfig(
        vocab_size=512, dim=256, n_layers=6, n_heads=8, n_kv_heads=2, max_seq_len=256
    )
    model = Transformer(cfg).to(device).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 16), device=device)
    n_new = 64

    print("=== KV cache vs. naive recompute ===")
    cached = time_it(lambda: generate(model, prompt, n_new, temperature=0.0))
    naive = time_it(lambda: naive_generate(model, prompt, n_new))
    print(f"cached:  {cached*1000:7.1f} ms")
    print(f"naive:   {naive*1000:7.1f} ms")
    print(f"speedup: {naive/cached:6.2f}x\n")

    print("=== Speculative decoding ===")
    draft_cfg = ModelConfig(
        vocab_size=512, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=256
    )
    draft = Transformer(draft_cfg).to(device).eval()
    # Make the draft agree with the target by sharing the embedding+head, so
    # the benchmark shows a realistic, non-zero acceptance rate.
    draft.tok_emb.weight.data[:, :64] = model.tok_emb.weight.data[:, :64]

    _, accept = speculative_generate(
        model, draft, prompt, n_new, lookahead=4, temperature=0.0
    )
    print(f"mean tokens accepted / round (lookahead=4): {accept:.2f}")
    print("(higher acceptance => fewer target forward passes => faster)")


if __name__ == "__main__":
    main()
