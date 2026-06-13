"""Tests for sampling and speculative decoding. The key correctness property:
speculative decoding with a draft == target must reproduce greedy decoding
exactly, since every proposed token is the target's own argmax."""

import torch
import pytest

from nanoformer import ModelConfig, Transformer, generate, speculative_generate
from nanoformer.generate import _filter_logits


@pytest.fixture
def model():
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=64, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64
    )
    return Transformer(cfg).eval()


def test_generate_extends_prompt(model):
    prompt = torch.randint(0, 64, (1, 5))
    out = generate(model, prompt, max_new_tokens=10, temperature=0.0)
    assert out.shape[1] == 15
    assert torch.equal(out[:, :5], prompt)


def test_greedy_is_deterministic(model):
    prompt = torch.randint(0, 64, (1, 5))
    a = generate(model, prompt, max_new_tokens=10, temperature=0.0)
    b = generate(model, prompt, max_new_tokens=10, temperature=0.0)
    assert torch.equal(a, b)


def test_eos_stops_generation(model):
    prompt = torch.randint(0, 64, (1, 5))
    # Force EOS to be whatever greedy picks first, then confirm truncation.
    out_full = generate(model, prompt, max_new_tokens=10, temperature=0.0)
    eos = out_full[0, 5].item()
    out = generate(model, prompt, max_new_tokens=10, temperature=0.0, eos_id=eos)
    assert out.shape[1] == 6  # prompt + single eos token


def test_respects_max_seq_len(model):
    prompt = torch.randint(0, 64, (1, 60))
    out = generate(model, prompt, max_new_tokens=20, temperature=0.0)
    assert out.shape[1] <= model.cfg.max_seq_len


def test_filter_logits_top_k():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    probs = _filter_logits(logits, temperature=1.0, top_k=2, top_p=None)
    assert (probs > 0).sum().item() == 2


def test_filter_logits_greedy():
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    probs = _filter_logits(logits, temperature=0.0, top_k=None, top_p=None)
    assert probs.argmax().item() == 1
    assert probs.max().item() == pytest.approx(1.0)


def test_speculative_equals_greedy(model):
    """draft == target, greedy: speculative output must equal plain greedy."""
    prompt = torch.randint(0, 64, (1, 5))
    greedy = generate(model, prompt, max_new_tokens=12, temperature=0.0)
    spec, accept = speculative_generate(
        model, model, prompt, max_new_tokens=12, lookahead=4, temperature=0.0
    )
    n = min(greedy.shape[1], spec.shape[1])
    assert torch.equal(greedy[:, :n], spec[:, :n])
    # Identical models => every draft token accepted => full lookahead.
    assert accept == pytest.approx(4.0)


def test_speculative_acceptance_rate_in_range(model):
    prompt = torch.randint(0, 64, (1, 5))
    _, accept = speculative_generate(
        model, model, prompt, max_new_tokens=8, lookahead=4, temperature=0.0
    )
    assert 0.0 <= accept <= 4.0
