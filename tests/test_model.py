"""Correctness tests for the model. The KV-cache equivalence test is the
most important one: it proves cached decoding produces identical logits to a
full forward pass, which is the contract the whole inference engine relies on.
"""

import torch
import pytest

from nanoformer import ModelConfig, Transformer
from nanoformer.model import RMSNorm, apply_rope, precompute_rope, repeat_kv


@pytest.fixture
def cfg():
    return ModelConfig(
        vocab_size=128, dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64
    )


def test_forward_shapes(cfg):
    model = Transformer(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss, _ = model(tokens)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss is None


def test_loss_is_finite(cfg):
    model = Transformer(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss, _ = model(tokens, targets=targets)
    assert torch.isfinite(loss)
    # Untrained loss should sit near ln(vocab_size).
    assert abs(loss.item() - torch.log(torch.tensor(float(cfg.vocab_size)))) < 1.0


def test_weight_tying(cfg):
    model = Transformer(cfg)
    assert model.lm_head.weight is model.tok_emb.weight


def test_rmsnorm_unit_scale():
    norm = RMSNorm(32)
    x = torch.randn(4, 10, 32) * 5.0
    out = norm(x)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rope_preserves_norm():
    cos, sin = precompute_rope(16, 32)
    x = torch.randn(1, 2, 8, 16)
    rotated = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-4)


def test_repeat_kv():
    x = torch.randn(1, 2, 5, 8)
    out = repeat_kv(x, 3)
    assert out.shape == (1, 6, 5, 8)
    # Heads should be interleaved-grouped, not arbitrary.
    assert torch.allclose(out[:, 0], out[:, 1])


def test_kv_cache_matches_full_forward(cfg):
    """Cached step-by-step decoding == one full forward pass (the core
    invariant of the inference engine)."""
    torch.manual_seed(0)
    model = Transformer(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 12))

    with torch.no_grad():
        full_logits, _, _ = model(tokens)

        cache = []
        step_logits = []
        for t in range(tokens.shape[1]):
            logits, _, cache = model(tokens[:, t : t + 1], past_kv=cache)
            step_logits.append(logits)
        cached = torch.cat(step_logits, dim=1)

    assert torch.allclose(full_logits, cached, atol=1e-4), (
        (full_logits - cached).abs().max().item()
    )


def test_prefill_then_decode_matches_full(cfg):
    """Prefilling a prompt then decoding one token matches a full pass."""
    torch.manual_seed(0)
    model = Transformer(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 10))
    with torch.no_grad():
        full, _, _ = model(tokens)
        prefill_logits, _, cache = model(tokens[:, :-1], past_kv=[])
        last, _, _ = model(tokens[:, -1:], past_kv=cache)
    assert torch.allclose(full[:, -1, :], last[:, -1, :], atol=1e-4)


def test_gqa_equals_mha_when_kv_equals_heads():
    cfg = ModelConfig(
        vocab_size=64, dim=32, n_layers=1, n_heads=4, n_kv_heads=4, max_seq_len=32
    )
    model = Transformer(cfg)
    assert model.blocks[0].attn.n_rep == 1


def test_param_count_reasonable(cfg):
    model = Transformer(cfg)
    n = model.num_parameters()
    assert n > 0
    # Weight tying: the embedding and lm_head share storage, so the head adds
    # no parameters. Recreate without tying and confirm it has strictly more.
    untied = Transformer(ModelConfig(**{**cfg.to_dict(), "tie_weights": False}))
    assert untied.num_parameters() == n + cfg.vocab_size * cfg.dim
