"""Tests for the BPE tokenizer: round-trip fidelity, vocab sizing, merge
behaviour, and persistence."""

import torch
import pytest

from nanoformer import BPETokenizer


SAMPLE = (
    "the quick brown fox jumps over the lazy dog. "
    "the dog was not amused, but the fox kept jumping. " * 20
)


@pytest.fixture
def tok():
    t = BPETokenizer()
    t.train(SAMPLE, vocab_size=400)
    return t


def test_roundtrip_is_lossless(tok):
    for s in ["the quick brown fox", "hello, world!", "unseen vocabulary 123"]:
        assert tok.decode(tok.encode(s)) == s


def test_roundtrip_unicode(tok):
    s = "café résumé naïve — emoji test 🚀"
    assert tok.decode(tok.encode(s)) == s


def test_vocab_size_respected():
    t = BPETokenizer()
    t.train(SAMPLE, vocab_size=300)
    assert t.vocab_size <= 300


def test_merges_reduce_length(tok):
    raw = list("the the the the".encode("utf-8"))
    encoded = tok.encode("the the the the")
    assert len(encoded) < len(raw)


def test_special_tokens(tok):
    ids = tok.encode("hi", bos=True, eos=True)
    assert ids[0] == tok.special["<|bos|>"]
    assert ids[-1] == tok.special["<|eos|>"]
    assert tok.decode(ids) == "<|bos|>hi<|eos|>"


def test_save_load(tok, tmp_path):
    p = tmp_path / "tok.json"
    tok.save(p)
    loaded = BPETokenizer.load(p)
    s = "the quick brown fox"
    assert loaded.encode(s) == tok.encode(s)


def test_encode_returns_valid_ids(tok):
    ids = tok.encode(SAMPLE[:200])
    assert all(0 <= i < tok.vocab_size for i in ids)
