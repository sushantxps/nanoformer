"""A byte-level BPE tokenizer, trained from scratch.

GPT-2-style approach: the corpus is split into word-like chunks with a
regex, BPE merges are learned over the (chunk -> frequency) table, and
encoding applies merges per chunk with an LRU cache. Working over unique
chunks instead of the raw byte stream makes training fast in pure Python.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

# Word-ish chunking. Keeping a leading space attached to a word lets the
# tokenizer learn tokens like " the", which roughly halves sequence length.
_SPLIT = re.compile(r" ?\w+| ?[^\w\s]+|\s+")


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self, text: str, vocab_size: int, special_tokens: tuple[str, ...] = ("<|bos|>", "<|eos|>")
    ) -> None:
        assert vocab_size >= 256 + len(special_tokens)
        n_merges = vocab_size - 256 - len(special_tokens)

        chunk_freq = Counter(_SPLIT.findall(text))
        # Each chunk starts as a tuple of byte ids.
        words: dict[tuple[int, ...], int] = {
            tuple(chunk.encode("utf-8")): freq for chunk, freq in chunk_freq.items()
        }

        for i in range(n_merges):
            pair_freq: Counter = Counter()
            for word, freq in words.items():
                for pair in zip(word, word[1:]):
                    pair_freq[pair] += freq
            if not pair_freq:
                break
            best = max(pair_freq, key=pair_freq.get)
            new_id = 256 + i
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            words = {self._merge_word(w, best, new_id): f for w, f in words.items()}

        for j, tok in enumerate(special_tokens):
            self.special[tok] = 256 + n_merges + j

        self._encode_chunk_cached.cache_clear()

    @staticmethod
    def _merge_word(
        word: tuple[int, ...], pair: tuple[int, int], new_id: int
    ) -> tuple[int, ...]:
        out: list[int] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    @lru_cache(maxsize=65536)
    def _encode_chunk_cached(self, chunk: str) -> tuple[int, ...]:
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            pairs = list(zip(ids, ids[1:]))
            # Merge the pair that was learned earliest (lowest merge id).
            candidate = min(
                (p for p in pairs if p in self.merges),
                key=lambda p: self.merges[p],
                default=None,
            )
            if candidate is None:
                break
            ids = list(self._merge_word(tuple(ids), candidate, self.merges[candidate]))
        return tuple(ids)

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        ids: list[int] = []
        if bos and "<|bos|>" in self.special:
            ids.append(self.special["<|bos|>"])
        for chunk in _SPLIT.findall(text):
            ids.extend(self._encode_chunk_cached(chunk))
        if eos and "<|eos|>" in self.special:
            ids.append(self.special["<|eos|>"])
        return ids

    def decode(self, ids: list[int]) -> str:
        inv_special = {v: k for k, v in self.special.items()}
        out = bytearray()
        for i in ids:
            if i in inv_special:
                out.extend(inv_special[i].encode("utf-8"))
            else:
                out.extend(self.vocab[i])
        return out.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        data = {
            "merges": [[a, b, idx] for (a, b), idx in self.merges.items()],
            "special": self.special,
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text())
        tok = cls()
        for a, b, idx in data["merges"]:
            tok.merges[(a, b)] = idx
            tok.vocab[idx] = tok.vocab[a] + tok.vocab[b]
        tok.special = dict(data["special"])
        return tok

    def __hash__(self) -> int:  # required for lru_cache on a method
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other
