"""Autoregressive generation: cached decoding, nucleus sampling, and a
self-contained implementation of speculative decoding.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .model import Transformer, KVCache


def _filter_logits(
    logits: torch.Tensor, temperature: float, top_k: Optional[int], top_p: Optional[float]
) -> torch.Tensor:
    """Apply temperature, then optional top-k and top-p (nucleus) filtering.

    Returns a probability distribution over the vocabulary.
    """
    if temperature <= 0:
        # Greedy: a one-hot distribution on the argmax.
        probs = torch.zeros_like(logits)
        probs.scatter_(-1, logits.argmax(-1, keepdim=True), 1.0)
        return probs

    logits = logits / temperature

    if top_k is not None:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum - F.softmax(sorted_logits, dim=-1) > top_p
        remove[..., 0] = False  # always keep the most probable token
        scatter_remove = remove.scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(scatter_remove, float("-inf"))

    return F.softmax(logits, dim=-1)


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """Standard cached autoregressive generation.

    Args:
        prompt: (1, prompt_len) token ids.
    Returns:
        (1, prompt_len + generated) token ids.
    """
    model.eval()
    device = next(model.parameters()).device
    tokens = prompt.to(device)

    # Prefill the cache with the whole prompt in one pass.
    logits, _, cache = model(tokens, past_kv=[])
    generated = []

    for _ in range(max_new_tokens):
        probs = _filter_logits(logits[:, -1, :], temperature, top_k, top_p)
        next_tok = torch.multinomial(probs, num_samples=1)
        generated.append(next_tok)
        if eos_id is not None and next_tok.item() == eos_id:
            break
        if tokens.shape[1] + len(generated) >= model.cfg.max_seq_len:
            break
        logits, _, cache = model(next_tok, past_kv=cache)

    if generated:
        tokens = torch.cat([tokens] + generated, dim=1)
    return tokens


@torch.no_grad()
def speculative_generate(
    target: Transformer,
    draft: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    lookahead: int = 4,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_id: Optional[int] = None,
) -> tuple[torch.Tensor, float]:
    """Speculative decoding (Leviathan et al., 2023).

    A small ``draft`` model proposes ``lookahead`` tokens; the large
    ``target`` model verifies them in a single forward pass and accepts the
    longest correct prefix. This preserves the target model's output
    distribution while cutting the number of expensive forward passes.

    Returns ``(tokens, acceptance_rate)`` where the acceptance rate is the
    mean number of draft tokens accepted per verification round, useful as a
    speedup proxy.
    """
    target.eval()
    draft.eval()
    device = next(target.parameters()).device
    tokens = prompt.to(device)

    # Prefill both models.
    t_logits, _, t_cache = target(tokens, past_kv=[])
    _, _, d_cache = draft(tokens, past_kv=[])

    n_generated = 0
    accepted_total = 0
    rounds = 0
    max_len = target.cfg.max_seq_len

    def sample(logits_row: torch.Tensor) -> torch.Tensor:
        probs = _filter_logits(logits_row, temperature, top_k, top_p)
        return torch.multinomial(probs, num_samples=1)

    while n_generated < max_new_tokens and tokens.shape[1] < max_len:
        rounds += 1
        # 1. Draft proposes `lookahead` tokens autoregressively.
        draft_tokens = []
        last = tokens[:, -1:]
        d_logits, _, d_cache = draft(last, past_kv=d_cache)
        for _ in range(lookahead):
            nxt = sample(d_logits[:, -1, :])
            draft_tokens.append(nxt)
            d_logits, _, d_cache = draft(nxt, past_kv=d_cache)
        draft_block = torch.cat(draft_tokens, dim=1)  # (1, lookahead)

        # 2. Target verifies all proposed tokens in one pass. We feed the
        #    last accepted token plus the draft block so the target produces
        #    a "next-token" distribution aligned with each draft position.
        verify_in = torch.cat([tokens[:, -1:], draft_block], dim=1)
        v_logits, _, t_cache = target(verify_in, past_kv=t_cache)

        # 3. Greedy acceptance test: accept a draft token while it matches
        #    the target's argmax at that position.
        n_accept = 0
        for i in range(lookahead):
            target_tok = v_logits[:, i, :].argmax(-1, keepdim=True)
            if target_tok.item() == draft_block[:, i].item():
                n_accept += 1
            else:
                break

        accepted_total += n_accept
        if n_accept == lookahead:
            # All accepted: also take the target's free bonus token.
            bonus = v_logits[:, lookahead, :].argmax(-1, keepdim=True)
            new_block = torch.cat([draft_block, bonus], dim=1)
        else:
            # First mismatch: replace with the target's own token and stop.
            correction = v_logits[:, n_accept, :].argmax(-1, keepdim=True)
            new_block = torch.cat([draft_block[:, :n_accept], correction], dim=1)

        # 4. Roll back both caches to the verified length and commit.
        committed = new_block.shape[1]
        tokens = torch.cat([tokens, new_block], dim=1)
        n_generated += committed
        keep = tokens.shape[1] - 1  # caches exclude the final un-attended token
        t_cache = target.crop_cache(t_cache, keep)
        d_cache = draft.crop_cache(d_cache, keep)

        if eos_id is not None and (new_block == eos_id).any():
            break

    acceptance_rate = accepted_total / max(rounds, 1)
    return tokens[:, : prompt.shape[1] + max_new_tokens], acceptance_rate
