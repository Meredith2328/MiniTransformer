# Scripts README

This folder now includes a small `cProfile` workflow for practicing tokenization profiling without touching the main training loop.

## Files

- `scripts/profile_tokenization.py`
  - Generates a `.prof` file for a chosen tokenization workload.
- `scripts/inspect_profile.py`
  - Reads a saved `.prof` file and prints stats, callers, or callees.
- `scripts/run_tokenization_profile_experiment.ps1`
  - Re-runs the fixed TinyStories profiling experiment used in the report below.
- `scripts/TOKENIZATION_PROFILING_EXPERIMENT.md`
  - Records one actual profiling run, the commands used, and the observed bottlenecks.
- `scripts/tokenize_to_bin.py`
  - Your normal tokenization entrypoint. The profiling script reuses its tokenizer-loading logic.

## Recommended workflow

### 1. Start from the tokenizer hot path

This isolates `Tokenizer.encode`, regex pretokenization, caching, and BPE merge application.

```powershell
uv run python scripts/profile_tokenization.py `
  --stage encode-sample `
  --input-text data/TinyStoriesV2-GPT4-train.txt `
  --vocab-pkl tokenizer/tinystories_bpe_vocab.pkl `
  --merges-pkl tokenizer/tinystories_bpe_merges.pkl `
  --max-lines 2000 `
  --repeat 5
```

Why this stage first:

- it minimizes disk I/O noise
- it makes tokenizer hotspots much easier to see
- `--repeat` amplifies the hot functions so the profile is easier to read

Output:

- a `.prof` file under `runs/profiles/tokenization/`
- a short on-screen preview sorted by cumulative time

## 2. Inspect the saved profile

Look at cumulative time first:

```powershell
uv run python scripts/inspect_profile.py `
  --profile runs/profiles/tokenization/<your-file>.prof `
  --sort cumulative `
  --top-k 40
```

Then inspect self time:

```powershell
uv run python scripts/inspect_profile.py `
  --profile runs/profiles/tokenization/<your-file>.prof `
  --sort tottime `
  --top-k 40
```

Practical interpretation:

- `cumulative` / `cumtime`: where total time is spent including child calls
- `tottime`: where the function body itself is expensive

For this tokenizer, likely hotspots are:

- `Tokenizer.encode`
- `Tokenizer._encode_plain_text`
- `Tokenizer._apply_bpe`
- regex `finditer`

## 3. Compare tokenizer work vs write work

If you want to separate encoding from binary writing, use different stages.

Count pass:

```powershell
uv run python scripts/profile_tokenization.py `
  --stage count-pass `
  --max-lines 5000
```

Write pass only:

```powershell
uv run python scripts/profile_tokenization.py `
  --stage write-pass `
  --max-lines 5000
```

Full limited pipeline:

```powershell
uv run python scripts/profile_tokenization.py `
  --stage full-pipeline `
  --max-lines 5000
```

How to read them:

- `count-pass` is mostly tokenizer compute
- `write-pass` isolates `np.memmap` write cost after tokens are already computed
- `full-pipeline` is the closest small-scale approximation to `tokenize_to_bin.py`

## 4. Use callers / callees when one function looks suspicious

Example: inspect who calls `_apply_bpe`.

```powershell
uv run python scripts/inspect_profile.py `
  --profile runs/profiles/tokenization/<your-file>.prof `
  --mode callers `
  --contains _apply_bpe
```

Or see what `_encode_plain_text` spends time inside:

```powershell
uv run python scripts/inspect_profile.py `
  --profile runs/profiles/tokenization/<your-file>.prof `
  --mode callees `
  --contains _encode_plain_text
```

## 5. A simple practice plan

Try this in order:

1. Run `encode-sample` on `2000` lines with `--repeat 5`.
2. Read the profile with `--sort cumulative`.
3. Read the same profile again with `--sort tottime`.
4. Run `write-pass` and compare whether writing is actually important.
5. Run `full-pipeline` and see whether the ranking changes.

This teaches the core distinction:

- pure tokenizer compute bottlenecks
- pipeline overhead
- write-path overhead

## Notes

- Keep `--max-lines` moderate at first. Profiling huge corpora produces large and noisy profiles.
- `write-pass` pre-encodes lines before the profile starts on purpose. That stage is for measuring write cost only.
- `full-pipeline` intentionally re-encodes lines during both count and write phases, matching the two-pass structure of `scripts/tokenize_to_bin.py`.
- If you later want line-level detail, that is a separate tool category such as `line_profiler`; start with `cProfile` first.
