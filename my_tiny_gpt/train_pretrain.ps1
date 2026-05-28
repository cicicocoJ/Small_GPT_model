# train_pretrain.ps1
# 用于集中调整 MiniGPT 预训练参数

$PYTHON_CMD = "python"
$TRAIN_SCRIPT = "train_pretrain.py"

# =========================
# 训练参数：主要改这里
# =========================

$CONTEXT_LENGTH = 128
$EMB_DIM = 128
$N_HEADS = 4
$N_LAYERS = 4
$DROPOUT = 0.1

$STRIDE = 64
$TRAIN_RATIO = 0.9
$BATCH_SIZE = 8
$EPOCHS = 30
$LEARNING_RATE = "3e-4"
$WEIGHT_DECAY = 0.01

$EVAL_FREQ = 5
$EVAL_BATCHES = 5

$MAX_NEW_TOKENS = 120
$PROMPT = "Learning"
$SEED = 123

# 如果想强制使用 CPU，把下面改成 $true
$USE_CPU = $false

Write-Host "开始预训练 MiniGPT..."
Write-Host "epochs: $EPOCHS"
Write-Host "batch_size: $BATCH_SIZE"
Write-Host "learning_rate: $LEARNING_RATE"
Write-Host "context_length: $CONTEXT_LENGTH"
Write-Host "emb_dim: $EMB_DIM"
Write-Host "n_layers: $N_LAYERS"
Write-Host "n_heads: $N_HEADS"

$ARGS = @(
    $TRAIN_SCRIPT,
    "--context_length", $CONTEXT_LENGTH,
    "--emb_dim", $EMB_DIM,
    "--n_heads", $N_HEADS,
    "--n_layers", $N_LAYERS,
    "--dropout", $DROPOUT,
    "--stride", $STRIDE,
    "--train_ratio", $TRAIN_RATIO,
    "--batch_size", $BATCH_SIZE,
    "--epochs", $EPOCHS,
    "--learning_rate", $LEARNING_RATE,
    "--weight_decay", $WEIGHT_DECAY,
    "--eval_freq", $EVAL_FREQ,
    "--eval_batches", $EVAL_BATCHES,
    "--max_new_tokens", $MAX_NEW_TOKENS,
    "--prompt", $PROMPT,
    "--seed", $SEED
)

if ($USE_CPU) {
    $ARGS += "--cpu"
}

& $PYTHON_CMD @ARGS

Write-Host "预训练完成。"
Write-Host "请查看 my_tiny_gpt/outputs/ 下的模型、loss 曲线和生成样例。"