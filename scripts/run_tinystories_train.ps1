param(
    [string]$CondaEnvPath = "C:\Software\Miniconda\envs\cs336",
    [string]$DataDir = "data",
    [string]$TokenizerDir = "tokenizer",
    [string]$RunsDir = "runs/tinystories_base",

    [string]$TrainTxt = "TinyStoriesV2-GPT4-train.txt",
    [string]$ValTxt = "TinyStoriesV2-GPT4-valid.txt",
    [int]$VocabSize = 10000,
    [string]$SpecialTokens = "<|endoftext|>",

    [int]$ContextLength = 256,
    [int]$DModel = 512,
    [int]$NumHeads = 16,
    [int]$DFF = 1344,
    [int]$NumLayers = 4,
    [double]$RopeTheta = 10000.0,

    [int]$BatchSize = 64,
    [int64]$TokenBudget = 327680000,
    [double]$LearningRate = 6e-4,
    [double]$MinLearningRate = 6e-5,
    [double]$WarmupFraction = 0.02,
    [double]$Beta1 = 0.9,
    [double]$Beta2 = 0.95,
    [double]$Eps = 1e-8,
    [double]$WeightDecay = 0.1,
    [double]$GradClip = 1.0,
    [string]$Optimizer = "custom_adamw",

    [int]$EvalInterval = 500,
    [int]$EvalIters = 50,
    [int]$LogInterval = 50,
    [int]$SaveInterval = 1000,
    [string]$Device = "auto",
    [int]$Seed = 1337,

    [switch]$UseWandb,
    [string]$WandbProject = "cs336-assignment1",
    [string]$WandbEntity = "",
    [string]$WandbRunName = "",
    [ValidateSet("online", "offline", "disabled")]
    [string]$WandbMode = "online",

    [switch]$SkipBpe,
    [switch]$SkipTokenize,
    [int]$BpeProgressEvery = 100,
    [int]$BpeHeartbeatSeconds = 15,
    [int]$TokenizeProgressEveryLines = 10000
)

$ErrorActionPreference = "Stop"

if (Test-Path "C:\Software\Miniconda\shell\condabin\conda-hook.ps1") {
    & C:\Software\Miniconda\shell\condabin\conda-hook.ps1
    conda activate $CondaEnvPath
}

$trainTxtPath = Join-Path $DataDir $TrainTxt
$valTxtPath = Join-Path $DataDir $ValTxt
if (-not (Test-Path $trainTxtPath)) {
    throw "Training txt not found: $trainTxtPath"
}
if (-not (Test-Path $valTxtPath)) {
    throw "Validation txt not found: $valTxtPath"
}

New-Item -ItemType Directory -Path $TokenizerDir -Force | Out-Null
New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
New-Item -ItemType Directory -Path $RunsDir -Force | Out-Null

$vocabPkl = Join-Path $TokenizerDir "tinystories_bpe_vocab.pkl"
$mergesPkl = Join-Path $TokenizerDir "tinystories_bpe_merges.pkl"
$trainBin = Join-Path $DataDir "tinystories_train.bin"
$valBin = Join-Path $DataDir "tinystories_val.bin"
$trainMeta = "$trainBin.meta.json"
$valMeta = "$valBin.meta.json"

if (-not $SkipBpe) {
    Write-Host "============================================================"
    Write-Host "Step 1/3: Train BPE tokenizer"
    & uv run python -m cs336_basics.train_bpe `
        --input-path $trainTxtPath `
        --vocab-size $VocabSize `
        --special-tokens $SpecialTokens `
        --progress-every $BpeProgressEvery `
        --heartbeat-seconds $BpeHeartbeatSeconds `
        --tokenizer-dir $TokenizerDir `
        --vocab-out $vocabPkl `
        --merges-out $mergesPkl
    if ($LASTEXITCODE -ne 0) { throw "BPE training failed." }
}

if (-not (Test-Path $vocabPkl) -or -not (Test-Path $mergesPkl)) {
    throw "Tokenizer artifacts missing: $vocabPkl / $mergesPkl"
}

if (-not $SkipTokenize) {
    Write-Host "============================================================"
    Write-Host "Step 2/3: Tokenize train/val txt to .bin"
    & uv run python scripts/tokenize_to_bin.py `
        --input-text $trainTxtPath `
        --vocab-pkl $vocabPkl `
        --merges-pkl $mergesPkl `
        --output-bin $trainBin `
        --output-meta $trainMeta `
        --special-tokens $SpecialTokens `
        --dtype uint16 `
        --progress-every-lines $TokenizeProgressEveryLines
    if ($LASTEXITCODE -ne 0) { throw "Train tokenization failed." }

    & uv run python scripts/tokenize_to_bin.py `
        --input-text $valTxtPath `
        --vocab-pkl $vocabPkl `
        --merges-pkl $mergesPkl `
        --output-bin $valBin `
        --output-meta $valMeta `
        --special-tokens $SpecialTokens `
        --dtype uint16 `
        --progress-every-lines $TokenizeProgressEveryLines
    if ($LASTEXITCODE -ne 0) { throw "Val tokenization failed." }
}

if (-not (Test-Path $trainBin) -or -not (Test-Path $valBin)) {
    throw "Tokenized data missing: $trainBin / $valBin"
}

$steps = [int][Math]::Ceiling($TokenBudget / ($BatchSize * $ContextLength))
$warmupIters = [int][Math]::Max(1, [Math]::Round($steps * $WarmupFraction))

if ([string]::IsNullOrWhiteSpace($WandbRunName)) {
    $WandbRunName = "tinystories_base_bs${BatchSize}_ctx${ContextLength}_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
}

Write-Host "============================================================"
Write-Host "Step 3/3: Train model"
Write-Host "  train_bin: $trainBin"
Write-Host "  val_bin:   $valBin"
Write-Host "  steps:     $steps"
Write-Host "  warmup:    $warmupIters"
Write-Host "  save_dir:  $RunsDir"

$trainArgs = @(
    "python", "-m", "cs336_basics.train",
    "--train-data", $trainBin,
    "--val-data", $valBin,
    "--data-dtype", "uint16",
    "--vocab-size", $VocabSize,
    "--context-length", $ContextLength,
    "--d-model", $DModel,
    "--num-heads", $NumHeads,
    "--d-ff", $DFF,
    "--num-layers", $NumLayers,
    "--rope-theta", $RopeTheta,
    "--optimizer", $Optimizer,
    "--batch-size", $BatchSize,
    "--total-iters", $steps,
    "--learning-rate", $LearningRate,
    "--min-learning-rate", $MinLearningRate,
    "--warmup-iters", $warmupIters,
    "--beta1", $Beta1,
    "--beta2", $Beta2,
    "--eps", $Eps,
    "--weight-decay", $WeightDecay,
    "--grad-clip", $GradClip,
    "--eval-interval", $EvalInterval,
    "--eval-iters", $EvalIters,
    "--log-interval", $LogInterval,
    "--save-interval", $SaveInterval,
    "--save-dir", $RunsDir,
    "--device", $Device,
    "--seed", $Seed
)

if ($UseWandb) {
    $trainArgs += @("--wandb", "--wandb-project", $WandbProject, "--wandb-run-name", $WandbRunName, "--wandb-mode", $WandbMode)
    if (-not [string]::IsNullOrWhiteSpace($WandbEntity)) {
        $trainArgs += @("--wandb-entity", $WandbEntity)
    }
}

& uv run @trainArgs
if ($LASTEXITCODE -ne 0) {
    throw "Training failed."
}

Write-Host "============================================================"
Write-Host "Done."
Write-Host "Tokenizer:"
Write-Host "  $vocabPkl"
Write-Host "  $mergesPkl"
Write-Host "Tokenized data:"
Write-Host "  $trainBin"
Write-Host "  $valBin"
Write-Host "Checkpoints:"
Write-Host "  $RunsDir"
