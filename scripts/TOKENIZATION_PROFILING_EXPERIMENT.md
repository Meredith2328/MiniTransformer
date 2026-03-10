# Tokenization Profiling Experiment

## Goal

Practice `cProfile` on the tokenization pipeline and answer a simple engineering question:

- where time is actually spent in tokenization
- whether write-out is a real bottleneck
- whether the tokenizer cache is doing meaningful work

## Environment

- Date: 2026-03-10
- Python: 3.12.0
- `uv`: 0.9.28
- CPU: 12th Gen Intel(R) Core(TM) i5-12500H
- OS: Windows workstation

Important note:

- absolute timings below are local-machine specific
- the function ranking and bottleneck structure are the useful part

## Dataset Sample

I used a bounded TinyStories sample instead of the full corpus:

- source file: `data/TinyStoriesV2-GPT4-valid.txt`
- sample rule: first `5000` lines via `--max-lines 5000`
- sample size: `699,925` characters
- sample size: `700,219` UTF-8 bytes

Why this sample size:

- big enough to make the profile stable
- small enough to run repeatedly in a few seconds
- based on the validation split, so it is easy to rerun without touching training artifacts

## Commands

The exact experiment can be rerun with:

```powershell
.\scripts\run_tokenization_profile_experiment.ps1
```

The runner executes these four workloads:

1. `encode-sample`
2. `count-pass`
3. `write-pass`
4. `full-pipeline`

The raw profile artifacts are written under `runs/profiles/tokenization/exp_20260310/`.

## Results Summary

Sample token count:

- `170,285` tokens on the 5000-line sample

Measured wall-clock results:

| Workload | What it measures | Wall time | Notes |
| --- | --- | ---: | --- |
| `encode-sample` | steady-state repeated encoding on the same sample | `1.524s` | `25,000` `encode()` calls from `5000 x repeat(5)` |
| `count-pass` | one token-count pass | `0.424s` | closest to the first pass in `tokenize_to_bin.py` |
| `write-pass` | pre-encode once, then isolate memmap write cost | `0.588s` | includes `0.541s` pre-encode + `0.045s` write path |
| `full-pipeline` | count pass + write pass with re-encoding | `0.866s` | closest small-scale approximation to the real script |

## Top Hotspots

### 1. `encode-sample`

Top cumulative-time functions:

- `bpe.py:_encode_plain_text` -> `1.402s` cumulative, `0.835s` self time
- `bpe.py:encode` -> `1.504s` cumulative
- `bpe.py:_encode_piece` -> `0.146s` cumulative
- `bpe.py:_apply_bpe` -> `0.129s` cumulative
- regex `finditer`, match `group`, and list `extend` all appear near the top

Interpretation:

- the dominant cost is not file I/O
- the dominant cost is text pretokenization plus per-piece bookkeeping inside `_encode_plain_text`
- BPE merge application matters, but it is not the first bottleneck on this sample

### 2. `count-pass`

Top cumulative-time functions:

- `bpe.py:_encode_plain_text` -> `0.397s`
- `bpe.py:_encode_piece` -> `0.131s`
- `bpe.py:_apply_bpe` -> `0.115s`

Interpretation:

- a normal one-pass token count is mostly tokenizer compute
- the first useful optimization target is still `_encode_plain_text`

### 3. `write-pass`

Top cumulative-time functions:

- `preencode_lines` -> `0.541s`
- `write_preencoded_tokens` -> `0.045s`
- `memmap.flush` -> `0.024s`

Interpretation:

- once token IDs are already available, writing them out is much cheaper than producing them
- on this sample, pure write-path work is material but secondary
- `flush()` is the single biggest write-side cost

### 4. `full-pipeline`

Top cumulative-time functions:

- `full_pipeline` -> `0.865s`
- tokenizer `encode` across both passes -> `0.820s`
- `count_tokens_in_memory` -> `0.532s`
- `write_tokens_in_memory` -> `0.332s`

Top self-time functions:

- `_encode_plain_text` -> `0.400s`
- dict `get` -> `0.086s`
- regex `group` -> `0.071s`
- list `extend` -> `0.066s`
- `_apply_bpe` -> `0.055s`

Interpretation:

- end-to-end, tokenizer compute still dominates the pipeline
- write-out is not free, but it is clearly not the primary bottleneck here

## Cache Observation

This was the most useful result of the exercise.

`encode-sample` ran `25,000` calls to `Tokenizer.encode`, but `_encode_piece` was only called `4,006` times.

`count-pass` ran only `5,000` calls to `Tokenizer.encode`, and `_encode_piece` was also called `4,006` times.

This means the repeated `encode-sample` run is hitting the tokenizer piece cache heavily:

- repeated text still pays for regex scanning and list/dict work inside `_encode_plain_text`
- but it does not keep recomputing BPE merges for the same pieces

So the cache is real and effective, but it does not remove the pretokenization overhead.

## Call-Graph Check

For `full-pipeline`, the callees of `_encode_plain_text` were dominated by:

- `_encode_piece`
- regex `finditer`
- regex `group`
- dict `get`
- list `extend`

This matches the ranking above and confirms that `_encode_plain_text` is a good top-level optimization target.

## Conclusions

1. The current tokenization bottleneck is tokenizer compute, especially `_encode_plain_text`, not binary write-out.
2. BPE application is important, but on this implementation the regex-and-bookkeeping layer is even more expensive.
3. `np.memmap` writing is relatively efficient; `flush()` is the main write-side cost worth remembering.
4. The piece cache is working and substantially reduces repeated BPE work on repeated text.

## If I Were Optimizing Next

I would inspect these directions first:

1. reduce work inside `_encode_plain_text`
2. reduce regex overhead or reduce the number of Python-level operations per matched piece
3. check whether batching writes further changes anything, but only after tokenizer compute is addressed
