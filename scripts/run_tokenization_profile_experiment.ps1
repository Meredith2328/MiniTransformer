param(
    [string]$CondaEnv = "C:\Software\Miniconda\envs\cs336",
    [string]$InputText = "data/TinyStoriesV2-GPT4-valid.txt",
    [string]$VocabPkl = "tokenizer/tinystories_bpe_vocab.pkl",
    [string]$MergesPkl = "tokenizer/tinystories_bpe_merges.pkl",
    [int]$MaxLines = 5000,
    [int]$Repeat = 5,
    [string]$OutputDir = "runs/profiles/tokenization/exp_20260310"
)

$ErrorActionPreference = "Stop"

function Invoke-Profile {
    param(
        [string]$Stage,
        [string]$ProfilePath
    )

    uv run python scripts/profile_tokenization.py `
        --stage $Stage `
        --input-text $InputText `
        --vocab-pkl $VocabPkl `
        --merges-pkl $MergesPkl `
        --max-lines $MaxLines `
        --repeat $Repeat `
        --top-k 12 `
        --output-prof $ProfilePath
}

function Write-Inspect {
    param(
        [string]$ProfilePath,
        [string]$Sort,
        [string]$OutputPath
    )

    uv run python scripts/inspect_profile.py `
        --profile $ProfilePath `
        --sort $Sort `
        --top-k 12 | Set-Content -Encoding utf8 $OutputPath
}

& "C:\Software\Miniconda\shell\condabin\conda-hook.ps1"
conda activate $CondaEnv

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$encodeProf = Join-Path $OutputDir "encode_sample.prof"
$countProf = Join-Path $OutputDir "count_pass.prof"
$writeProf = Join-Path $OutputDir "write_pass.prof"
$fullProf = Join-Path $OutputDir "full_pipeline.prof"

Write-Host "Running encode-sample profile..."
Invoke-Profile -Stage "encode-sample" -ProfilePath $encodeProf

Write-Host "Running count-pass profile..."
Invoke-Profile -Stage "count-pass" -ProfilePath $countProf

Write-Host "Running write-pass profile..."
Invoke-Profile -Stage "write-pass" -ProfilePath $writeProf

Write-Host "Running full-pipeline profile..."
Invoke-Profile -Stage "full-pipeline" -ProfilePath $fullProf

Write-Host "Writing derived stats..."
Write-Inspect -ProfilePath $encodeProf -Sort "tottime" -OutputPath (Join-Path $OutputDir "encode_sample_tottime.txt")
Write-Inspect -ProfilePath $fullProf -Sort "tottime" -OutputPath (Join-Path $OutputDir "full_pipeline_tottime.txt")
uv run python scripts/inspect_profile.py `
    --profile $fullProf `
    --mode callees `
    --contains "_encode_plain_text" | Set-Content -Encoding utf8 (Join-Path $OutputDir "full_pipeline_encode_plain_text_callees.txt")

Write-Host ""
Write-Host "Experiment artifacts written to: $OutputDir"
