param(
    [Parameter(Mandatory = $true)]
    [string]$TrainData,

    [string]$ValData = "",
    [ValidateSet("uint16", "uint32", "int32", "int64")]
    [string]$DataDtype = "uint16",

    [int]$VocabSize = 10000,
    [int]$ContextLength = 256,
    [int]$DModel = 512,
    [int]$NumHeads = 16,
    [int]$DFF = 1344,
    [int]$NumLayers = 4,
    [double]$RopeTheta = 10000.0,

    [int]$BatchSize = 64,
    [int64]$TokenBudget = 327680000,

    [double[]]$LearningRates = @(1e-4, 2e-4, 3e-4, 6e-4, 1e-3, 2e-3),
    [double]$MinLrRatio = 0.1,
    [double]$WarmupFraction = 0.02,

    [double]$Beta1 = 0.9,
    [double]$Beta2 = 0.95,
    [double]$Eps = 1e-8,
    [double]$WeightDecay = 0.1,
    [double]$GradClip = 1.0,

    [int]$EvalInterval = 500,
    [int]$EvalIters = 50,
    [int]$LogInterval = 50,
    [int]$SaveInterval = 1000,
    [string]$Device = "auto",
    [int]$Seed = 1337,

    [string]$SaveRoot = "runs/lr_sweep",

    [switch]$UseWandb,
    [string]$WandbProject = "cs336-assignment1",
    [string]$WandbEntity = "",
    [ValidateSet("online", "offline", "disabled")]
    [string]$WandbMode = "online",

    [string]$CondaEnvPath = "C:\Software\Miniconda\envs\cs336"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $TrainData)) {
    throw "Train data not found: $TrainData"
}
if ($ValData -and (-not (Test-Path $ValData))) {
    throw "Val data not found: $ValData"
}
if ($BatchSize -le 0 -or $ContextLength -le 0) {
    throw "BatchSize and ContextLength must be positive."
}
if ($LearningRates.Count -eq 0) {
    throw "LearningRates cannot be empty."
}

if (Test-Path "C:\Software\Miniconda\shell\condabin\conda-hook.ps1") {
    & C:\Software\Miniconda\shell\condabin\conda-hook.ps1
    conda activate $CondaEnvPath
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$sweepRoot = Join-Path $SaveRoot "lr_$timestamp"
New-Item -ItemType Directory -Path $sweepRoot -Force | Out-Null

$resultsCsv = Join-Path $sweepRoot "results.csv"
"run_name,learning_rate,min_learning_rate,batch_size,steps,warmup_iters,save_dir,exit_code,start_time,end_time,duration_sec" | Set-Content -Path $resultsCsv

$steps = [int][Math]::Ceiling($TokenBudget / ($BatchSize * $ContextLength))
$warmupIters = [int][Math]::Max(1, [Math]::Round($steps * $WarmupFraction))

Write-Host "LR sweep root: $sweepRoot"
Write-Host "Computed steps=$steps from token_budget=$TokenBudget, batch_size=$BatchSize, context_length=$ContextLength"

foreach ($lr in $LearningRates) {
    $safeLr = ($lr.ToString("G")).Replace(".", "p")
    $runName = "lr_${safeLr}_bs${BatchSize}_ctx${ContextLength}"
    $runDir = Join-Path $sweepRoot $runName
    New-Item -ItemType Directory -Path $runDir -Force | Out-Null

    $minLr = [double]($lr * $MinLrRatio)
    $start = Get-Date

    Write-Host "============================================================"
    Write-Host "Starting run: $runName"
    Write-Host "  lr=$lr min_lr=$minLr steps=$steps warmup=$warmupIters"

    $trainArgs = @(
        "python", "-m", "cs336_basics.train",
        "--train-data", $TrainData,
        "--data-dtype", $DataDtype,
        "--vocab-size", $VocabSize,
        "--context-length", $ContextLength,
        "--d-model", $DModel,
        "--num-heads", $NumHeads,
        "--d-ff", $DFF,
        "--num-layers", $NumLayers,
        "--rope-theta", $RopeTheta,
        "--batch-size", $BatchSize,
        "--total-iters", $steps,
        "--learning-rate", $lr,
        "--min-learning-rate", $minLr,
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
        "--save-dir", $runDir,
        "--device", $Device,
        "--seed", $Seed
    )

    if ($ValData) {
        $trainArgs += @("--val-data", $ValData)
    }

    if ($UseWandb) {
        $trainArgs += @("--wandb", "--wandb-project", $WandbProject, "--wandb-run-name", $runName, "--wandb-mode", $WandbMode)
        if ($WandbEntity) {
            $trainArgs += @("--wandb-entity", $WandbEntity)
        }
    }

    & uv run @trainArgs
    $exitCode = $LASTEXITCODE

    $end = Get-Date
    $durationSec = [int]($end - $start).TotalSeconds
    "$runName,$lr,$minLr,$BatchSize,$steps,$warmupIters,$runDir,$exitCode,$($start.ToString("s")),$($end.ToString("s")),$durationSec" | Add-Content -Path $resultsCsv

    if ($exitCode -ne 0) {
        Write-Warning "Run failed ($runName), exit_code=$exitCode. Continuing to next LR."
    }
}

Write-Host "LR sweep complete. Summary: $resultsCsv"
