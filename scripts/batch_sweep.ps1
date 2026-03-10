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

    [int[]]$BatchSizes = @(1, 8, 16, 32, 64, 128),
    [int64]$TokenBudget = 327680000,

    [double]$BaseLearningRate = 6e-4,
    [int]$BaseBatchSize = 64,
    [double]$MinLrRatio = 0.1,
    [double]$WarmupFraction = 0.02,
    [switch]$DisableLinearLrScaling,
    [string]$LrOverrides = "",

    [double]$Beta1 = 0.9,
    [double]$Beta2 = 0.95,
    [double]$Eps = 1e-8,
    [double]$WeightDecay = 0.1,
    [double]$GradClip = 1.0,

    [int]$EvalInterval = 500,
    [int]$EvalIters = 50,
    [int]$LogInterval = 50,
    [int]$SaveInterval = 1000,
    [int]$KeepLastCheckpoints = 3,
    [string]$Device = "auto",
    [int]$Seed = 1337,

    [string]$SaveRoot = "runs/batch_sweep",

    [switch]$UseWandb,
    [string]$WandbProject = "cs336-assignment1",
    [string]$WandbEntity = "",
    [ValidateSet("online", "offline", "disabled")]
    [string]$WandbMode = "online",

    [string]$CondaEnvPath = "C:\Software\Miniconda\envs\cs336"
)

$ErrorActionPreference = "Stop"
$wrapperScript = Join-Path $PSScriptRoot "run_tinystories_train.ps1"

function Parse-LrOverrides {
    param([string]$Raw)
    $map = @{}
    if ([string]::IsNullOrWhiteSpace($Raw)) {
        return $map
    }
    foreach ($item in ($Raw -split ",")) {
        $pair = $item.Trim()
        if (-not $pair) {
            continue
        }
        $parts = $pair -split "="
        if ($parts.Count -ne 2) {
            throw "Invalid LrOverrides entry '$pair'. Expected format like '64=6e-4'."
        }
        $bs = [int]$parts[0].Trim()
        $lr = [double]$parts[1].Trim()
        $map[$bs] = $lr
    }
    return $map
}

if (-not (Test-Path $TrainData)) {
    throw "Train data not found: $TrainData"
}
if ($ValData -and (-not (Test-Path $ValData))) {
    throw "Val data not found: $ValData"
}
if ($BatchSizes.Count -eq 0) {
    throw "BatchSizes cannot be empty."
}
if ($BaseBatchSize -le 0) {
    throw "BaseBatchSize must be positive."
}

$lrMap = Parse-LrOverrides -Raw $LrOverrides
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$sweepRoot = Join-Path $SaveRoot "batch_$timestamp"
New-Item -ItemType Directory -Path $sweepRoot -Force | Out-Null

$resultsCsv = Join-Path $sweepRoot "results.csv"
"run_name,batch_size,learning_rate,min_learning_rate,steps,warmup_iters,save_dir,exit_code,start_time,end_time,duration_sec" | Set-Content -Path $resultsCsv

Write-Host "Batch sweep root: $sweepRoot"
Write-Host "Token budget: $TokenBudget"

foreach ($batchSize in $BatchSizes) {
    if ($batchSize -le 0) {
        Write-Warning "Skipping invalid batch size: $batchSize"
        continue
    }

    if ($lrMap.ContainsKey($batchSize)) {
        $lr = [double]$lrMap[$batchSize]
    }
    elseif ($DisableLinearLrScaling) {
        $lr = $BaseLearningRate
    }
    else {
        $lr = [double]($BaseLearningRate * $batchSize / $BaseBatchSize)
    }

    $minLr = [double]($lr * $MinLrRatio)
    $steps = [int][Math]::Ceiling($TokenBudget / ($batchSize * $ContextLength))
    $warmupIters = [int][Math]::Max(1, [Math]::Round($steps * $WarmupFraction))

    $runName = "bs${batchSize}_lr$((($lr.ToString('G')).Replace('.', 'p')))_ctx${ContextLength}"
    $runDir = Join-Path $sweepRoot $runName
    New-Item -ItemType Directory -Path $runDir -Force | Out-Null

    $start = Get-Date
    Write-Host "============================================================"
    Write-Host "Starting run: $runName"
    Write-Host "  batch_size=$batchSize lr=$lr min_lr=$minLr steps=$steps warmup=$warmupIters"

    $wrapperParams = @{
        CondaEnvPath = $CondaEnvPath
        SkipBpe = $true
        SkipTokenize = $true
        TrainBin = $TrainData
        DataDtype = $DataDtype
        RunsDir = $runDir
        VocabSize = $VocabSize
        ContextLength = $ContextLength
        DModel = $DModel
        NumHeads = $NumHeads
        DFF = $DFF
        NumLayers = $NumLayers
        RopeTheta = $RopeTheta
        BatchSize = $batchSize
        TokenBudget = $TokenBudget
        LearningRate = $lr
        MinLearningRate = $minLr
        WarmupFraction = $WarmupFraction
        Beta1 = $Beta1
        Beta2 = $Beta2
        Eps = $Eps
        WeightDecay = $WeightDecay
        GradClip = $GradClip
        EvalInterval = $EvalInterval
        EvalIters = $EvalIters
        LogInterval = $LogInterval
        SaveInterval = $SaveInterval
        KeepLastCheckpoints = $KeepLastCheckpoints
        Device = $Device
        Seed = $Seed
    }

    if ($ValData) {
        $wrapperParams.ValBin = $ValData
    }
    else {
        $wrapperParams.ValBin = ""
        $wrapperParams.ValTxt = ""
    }

    if ($UseWandb) {
        $wrapperParams.UseWandb = $true
        $wrapperParams.WandbProject = $WandbProject
        $wrapperParams.WandbRunName = $runName
        $wrapperParams.WandbMode = $WandbMode
        if ($WandbEntity) {
            $wrapperParams.WandbEntity = $WandbEntity
        }
    }

    try {
        & $wrapperScript @wrapperParams
        $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
    }
    catch {
        $exitCode = 1
        Write-Warning "Run failed ($runName), exit_code=$exitCode. Continuing to next batch size."
        Write-Warning $_.Exception.Message
    }

    $end = Get-Date
    $durationSec = [int]($end - $start).TotalSeconds
    "$runName,$batchSize,$lr,$minLr,$steps,$warmupIters,$runDir,$exitCode,$($start.ToString("s")),$($end.ToString("s")),$durationSec" | Add-Content -Path $resultsCsv
}

Write-Host "Batch sweep complete. Summary: $resultsCsv"