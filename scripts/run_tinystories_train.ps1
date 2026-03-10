param(
    [string]$CondaEnvPath = "C:\Software\Miniconda\envs\cs336",
    [string]$DataDir = "data",
    [string]$TokenizerDir = "tokenizer",
    [string]$RunsDir = "runs/tinystories_base",

    [string]$TrainTxt = "TinyStoriesV2-GPT4-train.txt",
    [string]$ValTxt = "TinyStoriesV2-GPT4-valid.txt",
    [string]$TrainBin = "tinystories_train.bin",
    [string]$ValBin = "tinystories_val.bin",
    [int]$VocabSize = 10000,
    [string]$SpecialTokens = "<|endoftext|>",
    [ValidateSet("uint16", "uint32", "int32", "int64")]
    [string]$DataDtype = "uint16",

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
    [int]$KeepLastCheckpoints = 3,
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

function Resolve-PathArg {
    param(
        [string]$BaseDir,
        [string]$Value
    )
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return ""
    }
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return $Value
    }
    if ($Value.Contains([System.IO.Path]::DirectorySeparatorChar) -or $Value.Contains([System.IO.Path]::AltDirectorySeparatorChar)) {
        return $Value
    }
    return (Join-Path $BaseDir $Value)
}

function Require-File {
    param(
        [string]$Label,
        [string]$Path
    )
    if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path $Path)) {
        throw "Missing ${Label}: $Path"
    }
}

$trainBinProvided = $PSBoundParameters.ContainsKey("TrainBin")
$valBinProvided = $PSBoundParameters.ContainsKey("ValBin")
$valTxtProvided = $PSBoundParameters.ContainsKey("ValTxt")

if (Test-Path "C:\Software\Miniconda\shell\condabin\conda-hook.ps1") {
    & C:\Software\Miniconda\shell\condabin\conda-hook.ps1
    conda activate $CondaEnvPath
}

$trainTxtPath = Resolve-PathArg -BaseDir $DataDir -Value $TrainTxt
$valTxtPath = Resolve-PathArg -BaseDir $DataDir -Value $ValTxt
$TrainBin = Resolve-PathArg -BaseDir $DataDir -Value $TrainBin
$ValBin = Resolve-PathArg -BaseDir $DataDir -Value $ValBin

New-Item -ItemType Directory -Path $TokenizerDir -Force | Out-Null
New-Item -ItemType Directory -Path $RunsDir -Force | Out-Null
if (-not [string]::IsNullOrWhiteSpace($DataDir)) {
    New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
}
New-Item -ItemType Directory -Path (Split-Path -Parent $TrainBin) -Force | Out-Null

$useVal = $true
if (($valBinProvided -and [string]::IsNullOrWhiteSpace($ValBin)) -or ($valTxtProvided -and [string]::IsNullOrWhiteSpace($ValTxt))) {
    $useVal = $false
}
if ($useVal -and -not [string]::IsNullOrWhiteSpace($ValBin)) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $ValBin) -Force | Out-Null
}

$vocabPkl = Join-Path $TokenizerDir "tinystories_bpe_vocab.pkl"
$mergesPkl = Join-Path $TokenizerDir "tinystories_bpe_merges.pkl"
$trainMeta = "$TrainBin.meta.json"
$valMeta = "$ValBin.meta.json"

if (-not $SkipBpe) {
    if ([string]::IsNullOrWhiteSpace($trainTxtPath)) {
        throw "BPE training requires -TrainTxt."
    }
    Require-File -Label "train txt" -Path $trainTxtPath

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

if (-not $SkipTokenize) {
    if ([string]::IsNullOrWhiteSpace($trainTxtPath)) {
        throw "Tokenization requires -TrainTxt."
    }
    Require-File -Label "train txt" -Path $trainTxtPath
    Require-File -Label "tokenizer vocab" -Path $vocabPkl
    Require-File -Label "tokenizer merges" -Path $mergesPkl

    Write-Host "============================================================"
    Write-Host "Step 2/3: Tokenize train/val txt to .bin"
    & uv run python scripts/tokenize_to_bin.py `
        --input-text $trainTxtPath `
        --vocab-pkl $vocabPkl `
        --merges-pkl $mergesPkl `
        --output-bin $TrainBin `
        --output-meta $trainMeta `
        --special-tokens $SpecialTokens `
        --dtype $DataDtype `
        --progress-every-lines $TokenizeProgressEveryLines
    if ($LASTEXITCODE -ne 0) { throw "Train tokenization failed." }

    if ($useVal) {
        if ([string]::IsNullOrWhiteSpace($valTxtPath)) {
            throw "Validation tokenization requires -ValTxt, or disable validation with -ValBin ''."
        }
        Require-File -Label "val txt" -Path $valTxtPath

        & uv run python scripts/tokenize_to_bin.py `
            --input-text $valTxtPath `
            --vocab-pkl $vocabPkl `
            --merges-pkl $mergesPkl `
            --output-bin $ValBin `
            --output-meta $valMeta `
            --special-tokens $SpecialTokens `
            --dtype $DataDtype `
            --progress-every-lines $TokenizeProgressEveryLines
        if ($LASTEXITCODE -ne 0) { throw "Val tokenization failed." }
    }
}
else {
    Require-File -Label "train bin" -Path $TrainBin
    if ($useVal) {
        if (Test-Path $ValBin) {
            # keep validation enabled
        }
        elseif ($valBinProvided) {
            throw "Missing val bin: $ValBin"
        }
        else {
            Write-Host "Validation bin not found at $ValBin. Continuing without validation."
            $useVal = $false
            $ValBin = ""
        }
    }
}

Require-File -Label "train bin" -Path $TrainBin
if ($useVal) {
    Require-File -Label "val bin" -Path $ValBin
}

$steps = [int][Math]::Ceiling($TokenBudget / ($BatchSize * $ContextLength))
$warmupIters = [int][Math]::Max(1, [Math]::Round($steps * $WarmupFraction))

if ([string]::IsNullOrWhiteSpace($WandbRunName)) {
    $WandbRunName = "tinystories_base_bs${BatchSize}_ctx${ContextLength}_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
}

$valDisplay = if ($useVal) { $ValBin } else { "(disabled)" }

Write-Host "============================================================"
Write-Host "Step 3/3: Train model"
Write-Host "  train_bin: $TrainBin"
Write-Host "  val_bin:   $valDisplay"
Write-Host "  steps:     $steps"
Write-Host "  warmup:    $warmupIters"
Write-Host "  save_dir:  $RunsDir"

$trainArgs = @(
    "python", "-m", "cs336_basics.train",
    "--train-data", $TrainBin,
    "--data-dtype", $DataDtype,
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
    "--keep-last-checkpoints", $KeepLastCheckpoints,
    "--save-dir", $RunsDir,
    "--device", $Device,
    "--seed", $Seed
)

if ($useVal) {
    $trainArgs += @("--val-data", $ValBin)
}

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
Write-Host "  $TrainBin"
Write-Host "  $valDisplay"
Write-Host "Checkpoints:"
Write-Host "  $RunsDir"