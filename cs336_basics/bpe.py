from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
import json
import time
from typing import Any, BinaryIO, Callable
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
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        progress_every: int = 100,
    ) -> tuple[dict[int, bytes], list[Pair]]:
        def emit(event: dict[str, Any]) -> None:
            if progress_callback is not None:
                progress_callback(event)

        total_start = time.perf_counter()
        self._special_tokens = list(special_tokens)

        emit(
            {
                "stage": "read_corpus",
                "event": "start",
                "input_path": input_path,
            }
        )
        read_start = time.perf_counter()
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        emit(
            {
                "stage": "read_corpus",
                "event": "end",
                "seconds": time.perf_counter() - read_start,
                "num_chars": len(text),
            }
        )

        emit({"stage": "pretokenize", "event": "start"})
        pretoken_start = time.perf_counter()
        word_freq = self._pretokenize_and_count(text)
        emit(
            {
                "stage": "pretokenize",
                "event": "end",
                "seconds": time.perf_counter() - pretoken_start,
                "num_unique_words": len(word_freq),
                "num_word_instances": int(sum(word_freq.values())),
            }
        )

        emit({"stage": "init_vocab", "event": "start"})
        self._init_vocab(special_tokens)
        emit(
            {
                "stage": "init_vocab",
                "event": "end",
                "initial_vocab_size": len(self.vocab),
            }
        )

        merges_done = self._learn_merges(
            word_freq,
            vocab_size,
            progress_callback=progress_callback,
            progress_every=progress_every,
        )
        emit(
            {
                "stage": "train",
                "event": "end",
                "seconds": time.perf_counter() - total_start,
                "num_merges": merges_done,
                "final_vocab_size": len(self.vocab),
            }
        )
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

    def _learn_merges(
        self,
        word_freq: dict[Word, int],
        target_vocab_size: int,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        progress_every: int = 100,
    ) -> int:
        def emit(event: dict[str, Any]) -> None:
            if progress_callback is not None:
                progress_callback(event)

        remaining = max(0, target_vocab_size - len(self.vocab))
        pair_freq, pair_to_words = self._build_pair_stats(word_freq)
        progress_every = max(1, int(progress_every))
        merge_start = time.perf_counter()
        merges_done = 0

        emit(
            {
                "stage": "learn_merges",
                "event": "start",
                "target_merges": remaining,
                "starting_vocab_size": len(self.vocab),
            }
        )

        for merge_idx in range(remaining):
            if not pair_freq:
                break

            best_pair, best_count = max(pair_freq.items(), key=lambda item: (item[1], item[0]))
            if best_count <= 0:
                break

            self.merges.append(best_pair)
            self.vocab[len(self.vocab)] = best_pair[0] + best_pair[1]
            self._apply_merge(best_pair, word_freq, pair_freq, pair_to_words)
            merges_done = merge_idx + 1

            pair_freq.pop(best_pair, None)
            pair_to_words.pop(best_pair, None)

            if merges_done == 1 or merges_done % progress_every == 0 or merges_done == remaining:
                elapsed = time.perf_counter() - merge_start
                merge_rate = merges_done / elapsed if elapsed > 0 else 0.0
                pending = max(0, remaining - merges_done)
                eta_seconds = pending / merge_rate if merge_rate > 0 else None
                emit(
                    {
                        "stage": "learn_merges",
                        "event": "progress",
                        "completed_merges": merges_done,
                        "target_merges": remaining,
                        "progress": (merges_done / remaining) if remaining > 0 else 1.0,
                        "merge_rate_per_sec": merge_rate,
                        "eta_seconds": eta_seconds,
                        "current_pair_count": best_count,
                    }
                )

        emit(
            {
                "stage": "learn_merges",
                "event": "end",
                "completed_merges": merges_done,
                "target_merges": remaining,
                "seconds": time.perf_counter() - merge_start,
                "final_vocab_size": len(self.vocab),
            }
        )
        return merges_done

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


class Tokenizer:
    """Byte-level BPE tokenizer runtime."""

    _GPT2_PATTERN = ByteLevelBPE._GPT2_PATTERN

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges: list[tuple[bytes, bytes]] = list(merges)
        self.special_tokens: list[str] = list(special_tokens or [])
        self._single_byte_tokens: list[bytes] = [bytes([i]) for i in range(256)]

        self._append_missing_special_tokens()
        self.token_to_id: dict[bytes, int] = self._build_token_to_id()
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

        self._token_re = re.compile(self._GPT2_PATTERN)
        self._special_re = self._build_special_regex()
        self._piece_cache: dict[str, tuple[int, ...]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        if not isinstance(raw_vocab, dict):
            raise ValueError("Vocab file must contain a JSON object.")

        decoder = {v: k for k, v in cls._gpt2_bytes_to_unicode().items()}
        vocab: dict[int, bytes] = {}
        for token_text, token_id in raw_vocab.items():
            if not isinstance(token_id, int):
                raise ValueError("Vocab values must be integer token ids.")
            if not isinstance(token_text, str):
                raise ValueError("Vocab keys must be strings.")
            try:
                token_bytes = bytes(decoder[ch] for ch in token_text)
            except KeyError:
                token_bytes = token_text.encode("utf-8")
            vocab[token_id] = token_bytes

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                cleaned = line.rstrip("\r\n")
                if not cleaned:
                    continue
                parts = cleaned.split(" ")
                if len(parts) != 2:
                    continue
                left, right = parts
                try:
                    left_bytes = bytes(decoder[ch] for ch in left)
                    right_bytes = bytes(decoder[ch] for ch in right)
                except KeyError:
                    left_bytes = left.encode("utf-8")
                    right_bytes = right.encode("utf-8")
                merges.append((left_bytes, right_bytes))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        if self._special_re is None:
            return self._encode_plain_text(text)

        ids: list[int] = []
        cursor = 0
        for match in self._special_re.finditer(text):
            if match.start() > cursor:
                ids.extend(self._encode_plain_text(text[cursor : match.start()]))
            ids.append(self.token_to_id[match.group(0).encode("utf-8")])
            cursor = match.end()

        if cursor < len(text):
            ids.extend(self._encode_plain_text(text[cursor:]))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")

    def _append_missing_special_tokens(self) -> None:
        if not self.special_tokens:
            return

        existing_tokens = set(self.vocab.values())
        next_id = (max(self.vocab.keys()) + 1) if self.vocab else 0
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in existing_tokens:
                self.vocab[next_id] = token_bytes
                existing_tokens.add(token_bytes)
                next_id += 1

    def _build_token_to_id(self) -> dict[bytes, int]:
        token_to_id: dict[bytes, int] = {}
        for token_id in sorted(self.vocab):
            token_to_id.setdefault(self.vocab[token_id], token_id)
        return token_to_id

    def _build_special_regex(self) -> re.Pattern[str] | None:
        if not self.special_tokens:
            return None
        specials = sorted(self.special_tokens, key=len, reverse=True)
        escaped = "|".join(re.escape(token) for token in specials)
        return re.compile(f"(?:{escaped})")

    def _encode_plain_text(self, text: str) -> list[int]:
        ids: list[int] = []
        for match in self._token_re.finditer(text):
            piece = match.group(0)
            if not piece:
                continue

            cached = self._piece_cache.get(piece)
            if cached is None:
                cached = tuple(self._encode_piece(piece))
                self._piece_cache[piece] = cached
            ids.extend(cached)
        return ids

    def _encode_piece(self, piece: str) -> list[int]:
        word: Word = tuple(self._single_byte_tokens[b] for b in piece.encode("utf-8"))
        bpe_tokens = self._apply_bpe(word)
        return [self.token_to_id[token] for token in bpe_tokens]

    def _apply_bpe(self, word: Word) -> Word:
        if len(word) < 2:
            return word

        current = word
        while len(current) > 1:
            best_pair = None
            best_rank = None
            for i in range(len(current) - 1):
                pair = (current[i], current[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break
            current = ByteLevelBPE._merge_pair_in_word(current, best_pair)

        return current

    @staticmethod
    def _gpt2_bytes_to_unicode() -> dict[int, str]:
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        return {b: chr(c) for b, c in zip(bs, cs)}
