# Run Guide (TinyStories)

## 为什么训练用 `.bin` 而不是直接 `.txt`
- 训练脚本 [train.py](C:/desktoppp/cs336/assignment1-basics/cs336_basics/train.py) 采用 `np.memmap` + 随机切片采样，输入是**一维 token id 数组**，不是原始文本。
- 如果每个 step 都从 `.txt` 现场分词，会显著拖慢训练（尤其是你后面做大量 sweep 时）。
- 因此标准流程是：`txt -> tokenizer encode -> token ids .bin`，然后训练阶段只读 `.bin`。

这不是“额外负担”，而是你当前实现（和作业要求里的 memmap）对应的高效数据路径。

## 最简一键训练

在仓库根目录执行：

```powershell
& C:/Software/Miniconda/shell/condabin/conda-hook.ps1
conda activate C:/Software/Miniconda/envs/cs336
./scripts/run_tinystories_train.ps1 -UseWandb
```

这个脚本会自动做 3 件事：
1. 训练 BPE（如果没 `-SkipBpe`）
2. 把 `TinyStoriesV2-GPT4-train.txt / valid.txt` 转成 `.bin`（如果没 `-SkipTokenize`）
3. 启动模型训练

默认路径：
- 文本：`data/TinyStoriesV2-GPT4-train.txt`, `data/TinyStoriesV2-GPT4-valid.txt`
- tokenizer：`tokenizer/tinystories_bpe_vocab.pkl`, `tokenizer/tinystories_bpe_merges.pkl`
- 二进制数据：`data/tinystories_train.bin`, `data/tinystories_val.bin`
- 训练输出：`runs/tinystories_base`

## 只做 tokenization（单独跑）

```powershell
uv run python scripts/tokenize_to_bin.py `
  --input-text data/TinyStoriesV2-GPT4-train.txt `
  --vocab-pkl tokenizer/tinystories_bpe_vocab.pkl `
  --merges-pkl tokenizer/tinystories_bpe_merges.pkl `
  --output-bin data/tinystories_train.bin `
  --dtype uint16
```

验证集同理，把输入和输出路径改成 `valid` / `val` 即可。

## 单 minibatch 过拟合检查（建议先跑）

```powershell
uv run python scripts/overfit_single_batch.py `
  --train-data data/tinystories_train.bin `
  --data-dtype uint16 `
  --vocab-size 10000
```

如果 loss 明显下降，说明前向/反向/优化基本链路正常。

## 你当前作业参数的一键基线训练示例

```powershell
./scripts/run_tinystories_train.ps1 `
  -VocabSize 10000 `
  -ContextLength 256 `
  -DModel 512 `
  -NumHeads 16 `
  -DFF 1344 `
  -NumLayers 4 `
  -RopeTheta 10000 `
  -BatchSize 64 `
  -TokenBudget 327680000 `
  -LearningRate 6e-4 `
  -MinLearningRate 6e-5 `
  -UseWandb
```

脚本内部会自动计算 `total_iters = ceil(TokenBudget / (batch_size * context_length))`，并给出 warmup 步数。

## Sweep 脚本

### 学习率 sweep
```powershell
./scripts/lr_sweep.ps1 `
  -TrainData data/tinystories_train.bin `
  -ValData data/tinystories_val.bin `
  -VocabSize 10000 `
  -ContextLength 256 `
  -DModel 512 -NumHeads 16 -DFF 1344 -NumLayers 4 `
  -BatchSize 64 `
  -UseWandb
```

### Batch size sweep
```powershell
./scripts/batch_sweep.ps1 `
  -TrainData data/tinystories_train.bin `
  -ValData data/tinystories_val.bin `
  -VocabSize 10000 `
  -ContextLength 256 `
  -DModel 512 -NumHeads 16 -DFF 1344 -NumLayers 4 `
  -BatchSizes 1,8,16,32,64,128 `
  -UseWandb
```

两个 sweep 都会在对应 run 目录写 `results.csv`，便于你整理 experiment log。
