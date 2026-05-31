# train_pretrain.ps1
# TinyStories-only MiniGPT pretraining launcher.

param(
    [string]$Tokenizer = "regex",
    [string]$RunName = "tinystories_2k_fast",
    [switch]$Cpu,

    [string]$HfDataset = "roneneldan/TinyStories",
    [string]$HfTrainSplit = "train[:2000]",
    [string]$HfValSplit = "validation[:200]",
    [string]$HfLoadMode = "hub",
    [switch]$SaveHfToDisk,
    [string]$CacheDir = "data/hf_cache",
    [string]$HfDiskTrainPath = "data/TinyStories_train",
    [string]$HfDiskValPath = "data/TinyStories_val",

    [int]$Epochs = 10,
    [int]$ContextLength = 64,
    [int]$Stride = 64,
    [int]$EmbDim = 64,
    [int]$NLayers = 2,
    [int]$NHeads = 4,
    [double]$Dropout = 0.2,
    [int]$BatchSize = 16,
    [double]$LearningRate = 3e-4,
    [double]$WeightDecay = 0.05,
    [int]$EvalFreq = 50,
    [int]$EvalBatches = 100,

    [int]$MaxNewTokens = 150,
    [string]$Prompt = "Once upon a time",
    [double]$Temperature = 0.8,
    [int]$TopK = 40,
    [double]$RepetitionPenalty = 1.1,
    [int]$NoRepeatNgramSize = 3
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$cacheAbs = Join-Path $PSScriptRoot $CacheDir
$env:HF_HOME = $cacheAbs
$env:HF_HUB_CACHE = Join-Path $cacheAbs "hub"
$env:HF_DATASETS_CACHE = Join-Path $cacheAbs "datasets"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null
New-Item -ItemType Directory -Force -Path $env:HF_HUB_CACHE | Out-Null
New-Item -ItemType Directory -Force -Path $env:HF_DATASETS_CACHE | Out-Null

$device = if ($Cpu) { "cpu" } else { "auto cuda if available" }

Write-Host "MiniGPT TinyStories pretraining" -ForegroundColor Cyan
Write-Host "Working directory: $(Get-Location)"
Write-Host "HfDataset: $HfDataset"
Write-Host "HfTrainSplit: $HfTrainSplit"
Write-Host "HfValSplit: $HfValSplit"
Write-Host "HfLoadMode: $HfLoadMode"
Write-Host "CacheDir: $cacheAbs"
Write-Host "RunName: $RunName"
Write-Host "Tokenizer: $Tokenizer"
Write-Host "Epochs: $Epochs"
Write-Host "ContextLength: $ContextLength"
Write-Host "Stride: $Stride"
Write-Host "BatchSize: $BatchSize"
Write-Host "Device: $device"

$pythonArgs = @(
    "train_pretrain.py",
    "--hf_dataset", $HfDataset,
    "--hf_train_split", $HfTrainSplit,
    "--hf_val_split", $HfValSplit,
    "--hf_load_mode", $HfLoadMode,
    "--cache_dir", $CacheDir,
    "--hf_disk_train_path", $HfDiskTrainPath,
    "--hf_disk_val_path", $HfDiskValPath,
    "--tokenizer", $Tokenizer,
    "--save_best",
    "--collapse_blank_lines",
    "--sample",
    "--epochs", $Epochs,
    "--context_length", $ContextLength,
    "--stride", $Stride,
    "--emb_dim", $EmbDim,
    "--n_layers", $NLayers,
    "--n_heads", $NHeads,
    "--dropout", $Dropout,
    "--batch_size", $BatchSize,
    "--learning_rate", $LearningRate,
    "--weight_decay", $WeightDecay,
    "--eval_freq", $EvalFreq,
    "--eval_batches", $EvalBatches,
    "--early_stopping_patience", "10",
    "--min_delta", "0.0",
    "--max_new_tokens", $MaxNewTokens,
    "--prompt", $Prompt,
    "--temperature", $Temperature,
    "--top_k", $TopK,
    "--repetition_penalty", $RepetitionPenalty,
    "--no_repeat_ngram_size", $NoRepeatNgramSize,
    "--clean_text"
)

if ($RunName -ne "") {
    $pythonArgs += @("--run_name", $RunName)
}

if ($Cpu) {
    $pythonArgs += "--cpu"
}

if ($SaveHfToDisk) {
    $pythonArgs += "--save_hf_to_disk"
}

Write-Host "Command:" -ForegroundColor Cyan
Write-Host ("python " + ($pythonArgs -join " "))

& python @pythonArgs

if ($LASTEXITCODE -ne 0) {
    throw "Pretraining failed. Please check the error message above."
}

Write-Host "Pretraining finished." -ForegroundColor Green
