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
    [int]$KeepLastCheckpoints = 3,
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
$wrapperScript = Join-Path $PSScriptRoot "run_tinystories_train.ps1"

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
        BatchSize = $BatchSize
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
        Write-Warning "Run failed ($runName), exit_code=$exitCode. Continuing to next LR."
        Write-Warning $_.Exception.Message
    }

    $end = Get-Date
    $durationSec = [int]($end - $start).TotalSeconds
    "$runName,$lr,$minLr,$BatchSize,$steps,$warmupIters,$runDir,$exitCode,$($start.ToString("s")),$($end.ToString("s")),$durationSec" | Add-Content -Path $resultsCsv
}

Write-Host "LR sweep complete. Summary: $resultsCsv"