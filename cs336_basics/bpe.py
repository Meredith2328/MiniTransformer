from __future__ import annotations

from collections import defaultdict
from typing import BinaryIO
import os

import regex as re


Token = bytes
Word = tuple[Token, ...]
Pair = tuple[Token, Token]


class ByteLevelBPE:
    """Byte-level BPE tokenizer."""

    _GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self) -> None:
        self.vocab: dict[int, bytes] = {}
        self.merges: list[Pair] = []
        self._special_tokens: list[str] = []
        self._single_byte_tokens: list[bytes] = [bytes([i]) for i in range(256)]

    @classmethod
    def _get_gpt2_pattern(cls) -> str:
        return cls._GPT2_PATTERN

    @staticmethod
    def _find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int) -> list[int]:
        """Split a file by byte size into roughly equal chunks."""
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        if file_size == 0 or desired_num_chunks <= 1:
            return [0, file_size]

        chunk_size = file_size // desired_num_chunks
        boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        boundaries[-1] = file_size
        return boundaries

    def train(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
    ) -> tuple[dict[int, bytes], list[Pair]]:
        self._special_tokens = list(special_tokens)

        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        word_freq = self._pretokenize_and_count(text)
        self._init_vocab(special_tokens)
        self._learn_merges(word_freq, vocab_size)
        return self.vocab, self.merges

    def encode(self, text: str) -> list[int]:
        if not self.vocab:
            raise ValueError("Tokenizer not trained. Call train() first.")

        token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        special_set = set(self._special_tokens)
        output: list[int] = []

        for piece in self._split_into_tokens(text):
            if piece in special_set:
                output.append(token_to_id[piece.encode("utf-8")])
                continue

            word = tuple(self._single_byte_tokens[b] for b in piece.encode("utf-8"))
            for pair in self.merges:
                word = self._merge_pair_in_word(word, pair)

            for token in word:
                output.append(token_to_id[token])

        return output

    def decode(self, token_ids: list[int]) -> str:
        if not self.vocab:
            raise ValueError("Tokenizer not trained. Call train() first.")

        byte_sequence = b"".join(self.vocab[token_id] for token_id in token_ids)
        return byte_sequence.decode("utf-8", errors="replace")

    def _build_pretoken_regex(self, include_special_tokens: bool) -> re.Pattern[str]:
        if include_special_tokens and self._special_tokens:
            specials = sorted(self._special_tokens, key=len, reverse=True)
            escaped = "|".join(re.escape(token) for token in specials)
            return re.compile(f"(?:{escaped})|{self._GPT2_PATTERN}")
        return re.compile(self._GPT2_PATTERN)

    def _build_special_split_regex(self) -> re.Pattern[str] | None:
        if not self._special_tokens:
            return None
        specials = sorted(self._special_tokens, key=len, reverse=True)
        escaped = "|".join(re.escape(token) for token in specials)
        return re.compile(f"(?:{escaped})")

    def _pretokenize_and_count(self, text: str) -> dict[Word, int]:
        token_re = self._build_pretoken_regex(include_special_tokens=False)
        freq: dict[Word, int] = defaultdict(int)

        split_re = self._build_special_split_regex()
        segments = split_re.split(text) if split_re is not None else [text]

        # Special tokens split the corpus into independent spans and are not
        # part of merge statistics.
        for segment in segments:
            if not segment:
                continue
            for match in token_re.finditer(segment):
                piece = match.group(0)
                if not piece:
                    continue
                encoded = piece.encode("utf-8")
                word = tuple(self._single_byte_tokens[b] for b in encoded)
                freq[word] += 1

        return dict(freq)

    def _split_into_tokens(self, text: str) -> list[str]:
        token_re = self._build_pretoken_regex(include_special_tokens=True)
        return [m.group(0) for m in token_re.finditer(text)]

    # Compatibility with some local tests.
    def _pretokenize(self, text: str) -> list[str]:
        return self._split_into_tokens(text)

    def _init_vocab(self, special_tokens: list[str]) -> None:
        self.vocab = {}
        self.merges = []

        for i, token in enumerate(special_tokens):
            self.vocab[i] = token.encode("utf-8")

        start = len(special_tokens)
        for i in range(256):
            self.vocab[start + i] = self._single_byte_tokens[i]

    def _learn_merges(self, word_freq: dict[Word, int], target_vocab_size: int) -> None:
        remaining = max(0, target_vocab_size - len(self.vocab))
        pair_freq, pair_to_words = self._build_pair_stats(word_freq)

        for _ in range(remaining):
            if not pair_freq:
                break

            best_pair, best_count = max(pair_freq.items(), key=lambda item: (item[1], item[0]))
            if best_count <= 0:
                break

            self.merges.append(best_pair)
            self.vocab[len(self.vocab)] = best_pair[0] + best_pair[1]
            self._apply_merge(best_pair, word_freq, pair_freq, pair_to_words)

            pair_freq.pop(best_pair, None)
            pair_to_words.pop(best_pair, None)

    def _build_pair_stats(
        self,
        word_freq: dict[Word, int],
    ) -> tuple[dict[Pair, int], dict[Pair, set[Word]]]:
        pair_freq: dict[Pair, int] = defaultdict(int)
        pair_to_words: dict[Pair, set[Word]] = defaultdict(set)

        for word, count in word_freq.items():
            if len(word) < 2:
                continue
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freq[pair] += count
                pair_to_words[pair].add(word)

        return dict(pair_freq), dict(pair_to_words)

    def _apply_merge(
        self,
        pair: Pair,
        word_freq: dict[Word, int],
        pair_freq: dict[Pair, int],
        pair_to_words: dict[Pair, set[Word]],
    ) -> None:
        affected_words = list(pair_to_words.get(pair, set()))

        for old_word in affected_words:
            count = word_freq.get(old_word, 0)
            if count == 0:
                continue

            new_word = self._merge_pair_in_word(old_word, pair)
            if new_word == old_word:
                continue

            old_pair_counts = self._count_pairs(old_word)
            new_pair_counts = self._count_pairs(new_word)

            for p, occurrences in old_pair_counts.items():
                pair_freq[p] = pair_freq.get(p, 0) - occurrences * count
                if pair_freq[p] <= 0:
                    pair_freq.pop(p, None)
                if p in pair_to_words:
                    pair_to_words[p].discard(old_word)
                    if not pair_to_words[p]:
                        pair_to_words.pop(p, None)

            for p, occurrences in new_pair_counts.items():
                pair_freq[p] = pair_freq.get(p, 0) + occurrences * count
                pair_to_words.setdefault(p, set()).add(new_word)

            del word_freq[old_word]
            word_freq[new_word] = word_freq.get(new_word, 0) + count

    def _count_pairs(self, word: Word) -> dict[Pair, int]:
        counts: dict[Pair, int] = defaultdict(int)
        for i in range(len(word) - 1):
            counts[(word[i], word[i + 1])] += 1
        return dict(counts)

    @staticmethod
    def _merge_pair_in_word(word: Word, pair: Pair) -> Word:
        if len(word) < 2:
            return word

        first, second = pair
        merged = first + second
        out: list[bytes] = []
        i = 0
        n = len(word)
        while i < n:
            if i + 1 < n and word[i] == first and word[i + 1] == second:
                out.append(merged)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)
