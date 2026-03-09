from __future__ import annotations

import json
import os

import tiktoken

from .adapters import get_tokenizer
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"


def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path, encoding="utf-8") as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))

    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }

    if special_tokens:
        for special_token in special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in set(vocab.values()):
                vocab[len(vocab)] = token_bytes

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return get_tokenizer(vocab, merges, special_tokens)


def test_roundtrip_empty():
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH)
    test_string = ""
    assert tokenizer.decode(tokenizer.encode(test_string)) == test_string


def test_empty_matches_tiktoken():
    reference = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH)
    test_string = ""
    reference_ids = reference.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids
    assert tokenizer.decode(ids) == test_string


def test_roundtrip_single_character():
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH)
    test_string = "s"
    assert tokenizer.decode(tokenizer.encode(test_string)) == test_string


def test_single_character_matches_tiktoken():
    reference = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH)
    test_string = "s"
    reference_ids = reference.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids
    assert [tokenizer.decode([x]) for x in ids] == ["s"]


def test_roundtrip_single_unicode_character():
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH)
    test_string = "😁"
    assert tokenizer.decode(tokenizer.encode(test_string)) == test_string


def test_single_unicode_character_matches_tiktoken():
    reference = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH)
    test_string = "😁"
    reference_ids = reference.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids


def test_roundtrip_ascii_string():
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH)
    test_string = "Hello, how are you?"
    assert tokenizer.decode(tokenizer.encode(test_string)) == test_string


def test_ascii_string_matches_tiktoken():
    reference = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Hello, how are you?"
    reference_ids = reference.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids
    assert [tokenizer.decode([x]) for x in ids] == ["Hello", ",", " how", " are", " you", "?"]


def test_roundtrip_unicode_string():
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH)
    test_string = "Héllò hôw are ü? 😁"
    assert tokenizer.decode(tokenizer.encode(test_string)) == test_string


def test_unicode_string_matches_tiktoken():
    reference = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw are ü? 😁"
    reference_ids = reference.encode(test_string)
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids


def test_roundtrip_unicode_string_with_special_tokens():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 😁<|endoftext|>"
    encoded_ids = tokenizer.encode(test_string)
    tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
    assert tokenized_string.count("<|endoftext|>") == 3
    assert tokenizer.decode(encoded_ids) == test_string


def test_unicode_string_with_special_tokens_matches_tiktoken():
    reference = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 😁<|endoftext|>"
    reference_ids = reference.encode(test_string, allowed_special={"<|endoftext|>"})
    ids = tokenizer.encode(test_string)
    assert ids == reference_ids


def test_overlapping_special_tokens():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        VOCAB_PATH,
        MERGES_PATH,
        special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"],
    )
    test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    ids = tokenizer.encode(test_string)
    tokenized_string = [tokenizer.decode([x]) for x in ids]
    assert tokenized_string.count("<|endoftext|>") == 1
    assert tokenized_string.count("<|endoftext|><|endoftext|>") == 1
    assert tokenizer.decode(ids) == test_string


def test_address_roundtrip():
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH)
    with open(FIXTURES_PATH / "address.txt", encoding="utf-8") as f:
        corpus_contents = f.read()
    ids = tokenizer.encode(corpus_contents)
    assert tokenizer.decode(ids) == corpus_contents


def test_address_matches_tiktoken():
    reference = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH)
    with open(FIXTURES_PATH / "address.txt", encoding="utf-8") as f:
        corpus_contents = f.read()
    reference_ids = reference.encode(corpus_contents)
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids


def test_german_roundtrip():
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH)
    with open(FIXTURES_PATH / "german.txt", encoding="utf-8") as f:
        corpus_contents = f.read()
    ids = tokenizer.encode(corpus_contents)
    assert tokenizer.decode(ids) == corpus_contents


def test_german_matches_tiktoken():
    reference = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH)
    with open(FIXTURES_PATH / "german.txt", encoding="utf-8") as f:
        corpus_contents = f.read()
    reference_ids = reference.encode(corpus_contents)
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids


def test_tinystories_sample_roundtrip():
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH)
    with open(FIXTURES_PATH / "tinystories_sample.txt", encoding="utf-8") as f:
        corpus_contents = f.read()
    ids = tokenizer.encode(corpus_contents)
    assert tokenizer.decode(ids) == corpus_contents


def test_tinystories_matches_tiktoken():
    reference = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    with open(FIXTURES_PATH / "tinystories_sample.txt", encoding="utf-8") as f:
        corpus_contents = f.read()
    reference_ids = reference.encode(corpus_contents, allowed_special={"<|endoftext|>"})
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids


def test_encode_special_token_trailing_newlines():
    reference = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    with open(FIXTURES_PATH / "special_token_trailing_newlines.txt", encoding="utf-8") as f:
        corpus_contents = f.read()
    reference_ids = reference.encode(corpus_contents, allowed_special={"<|endoftext|>"})
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids


def test_encode_special_token_double_newline_non_whitespace():
    reference = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    with open(
        FIXTURES_PATH / "special_token_double_newlines_non_whitespace.txt",
        encoding="utf-8",
    ) as f:
        corpus_contents = f.read()
    reference_ids = reference.encode(corpus_contents, allowed_special={"<|endoftext|>"})
    ids = tokenizer.encode(corpus_contents)
    assert ids == reference_ids


def test_encode_iterable_tinystories_sample_roundtrip():
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH, MERGES_PATH)
    all_ids = []
    with open(FIXTURES_PATH / "tinystories_sample.txt", encoding="utf-8") as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)
    with open(FIXTURES_PATH / "tinystories_sample.txt", encoding="utf-8") as f:
        corpus_contents = f.read()
    assert tokenizer.decode(all_ids) == corpus_contents


def test_encode_iterable_tinystories_matches_tiktoken():
    reference = tiktoken.get_encoding("gpt2")
    tokenizer = get_tokenizer_from_vocab_merges_path(
        VOCAB_PATH, MERGES_PATH, special_tokens=["<|endoftext|>"]
    )
    with open(FIXTURES_PATH / "tinystories_sample.txt", encoding="utf-8") as f:
        corpus_contents = f.read()
    reference_ids = reference.encode(corpus_contents, allowed_special={"<|endoftext|>"})
    all_ids = []
    with open(FIXTURES_PATH / "tinystories_sample.txt", encoding="utf-8") as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)
    assert all_ids == reference_ids
