"""小型 GPT 预训练脚本。

运行方式（Windows VS Code 终端）：
    python .\\my_tiny_gpt\\train_pretrain.py

本脚本完成：
1. 读取 data/pretrain.txt；
2. 使用字符级 tokenizer 编码文本；
3. 用滑动窗口构造 next-token prediction 数据；
4. 训练 MiniGPT；
5. 保存模型、loss、日志和生成样例到 outputs/。
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mini_gpt import MiniGPT, MiniGPTConfig, count_parameters
from tiny_tokenizer import CharTokenizer


class PretrainDataset(Dataset):
    """预训练数据集。

    给定一串 token ids，用滑动窗口构造训练样本：
    input_ids = tokens[i : i + context_length]
    target_ids = tokens[i + 1 : i + context_length + 1]

    这就是语言模型的核心训练目标：根据前文预测下一个 token。
    """

    def __init__(self, token_ids: list[int], context_length: int, stride: int):
        if len(token_ids) <= context_length:
            raise ValueError("语料 token 数量必须大于 context_length")

        self.input_ids = []
        self.target_ids = []

        for start in range(0, len(token_ids) - context_length, stride):
            input_chunk = token_ids[start : start + context_length]
            target_chunk = token_ids[start + 1 : start + context_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.target_ids[index]


def split_token_ids(token_ids: list[int], train_ratio: float) -> tuple[list[int], list[int]]:
    """按比例切分训练集和验证集。"""

    split_idx = int(len(token_ids) * train_ratio)
    return token_ids[:split_idx], token_ids[split_idx:]


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: MiniGPT,
    device: torch.device,
) -> torch.Tensor:
    """计算一个 batch 的 next-token prediction 交叉熵损失。"""

    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_batch.reshape(-1),
    )
    return loss


def evaluate_loss(
    model: MiniGPT,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int = 5,
) -> float:
    """在训练集或验证集上估计平均 loss。"""

    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            num_batches += 1
            if num_batches >= max_batches:
                break

    model.train()
    if num_batches == 0:
        return float("nan")
    return total_loss / num_batches


def generate_text(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    """用贪心解码生成一小段文本，便于观察预训练效果。"""

    model.eval()
    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        # 超过上下文长度时，只保留最近的 context_length 个 token。
        input_cond = input_ids[:, -model.cfg.context_length :]

        with torch.no_grad():
            logits = model(input_cond)

        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token_id], dim=1)

    generated_ids = input_ids.squeeze(0).tolist()
    model.train()
    return tokenizer.decode(generated_ids)


def save_loss_csv(loss_rows: list[dict[str, float]], csv_path: Path) -> None:
    """保存 loss 记录，方便报告画表或复现实验。"""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "step", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(loss_rows)


def save_loss_plot(loss_rows: list[dict[str, float]], png_path: Path) -> None:
    """保存 loss 曲线图。

    如果环境没有 matplotlib，不影响训练，只跳过画图。
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过 loss 曲线图保存。")
        return

    steps = [row["step"] for row in loss_rows]
    train_losses = [row["train_loss"] for row in loss_rows]
    val_losses = [row["val_loss"] for row in loss_rows]

    plt.figure(figsize=(6, 4))
    plt.plot(steps, train_losses, label="train loss")
    plt.plot(steps, val_losses, label="val loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("MiniGPT Pretraining Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()


def train(args: argparse.Namespace) -> None:
    """主训练流程。"""

    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "data" / "pretrain.txt"
    output_dir = project_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = output_dir / "vocab.json"
    model_path = output_dir / "pretrain_model.pt"
    loss_csv_path = output_dir / "pretrain_loss.csv"
    loss_png_path = output_dir / "pretrain_loss.png"
    log_path = output_dir / "pretrain_log.txt"
    sample_path = output_dir / "pretrain_sample.txt"

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    text = data_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.build_from_text(text)
    tokenizer.save(vocab_path)
    token_ids = tokenizer.encode(text)

    train_ids, val_ids = split_token_ids(token_ids, args.train_ratio)
    train_dataset = PretrainDataset(train_ids, args.context_length, args.stride)
    val_dataset = PretrainDataset(val_ids, args.context_length, args.stride)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    cfg = MiniGPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=args.context_length,
        emb_dim=args.emb_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    model = MiniGPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    start_time = time.time()
    global_step = 0
    loss_rows = []
    log_lines = [
        "MiniGPT pretraining log",
        f"device: {device}",
        f"text chars: {len(text)}",
        f"tokens: {len(token_ids)}",
        f"vocab_size: {tokenizer.vocab_size}",
        f"train samples: {len(train_dataset)}",
        f"val samples: {len(val_dataset)}",
        f"parameters: {count_parameters(model)}",
        f"config: {json.dumps(cfg.__dict__, ensure_ascii=False)}",
    ]

    print("\n".join(log_lines))

    for epoch in range(1, args.epochs + 1):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % args.eval_freq == 0:
                train_loss = evaluate_loss(model, train_loader, device, args.eval_batches)
                val_loss = evaluate_loss(model, val_loader, device, args.eval_batches)
                row = {
                    "epoch": epoch,
                    "step": global_step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
                loss_rows.append(row)
                line = (
                    f"epoch {epoch:02d} step {global_step:04d} | "
                    f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f}"
                )
                print(line)
                log_lines.append(line)

    final_train_loss = evaluate_loss(model, train_loader, device, args.eval_batches)
    final_val_loss = evaluate_loss(model, val_loader, device, args.eval_batches)
    final_row = {
        "epoch": args.epochs,
        "step": global_step,
        "train_loss": final_train_loss,
        "val_loss": final_val_loss,
    }
    loss_rows.append(final_row)

    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "vocab_path": str(vocab_path),
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
    }
    torch.save(checkpoint, model_path)
    save_loss_csv(loss_rows, loss_csv_path)
    save_loss_plot(loss_rows, loss_png_path)
    sample_path.write_text(generated, encoding="utf-8")

    elapsed = time.time() - start_time
    log_lines.extend(
        [
            f"final_train_loss: {final_train_loss:.4f}",
            f"final_val_loss: {final_val_loss:.4f}",
            f"elapsed_seconds: {elapsed:.2f}",
            f"model_path: {model_path}",
            f"loss_csv_path: {loss_csv_path}",
            f"sample_path: {sample_path}",
            "generated sample:",
            generated,
        ]
    )
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    print("训练完成。")
    print(f"模型已保存: {model_path}")
    print(f"loss 记录已保存: {loss_csv_path}")
    print(f"生成样例已保存: {sample_path}")
    print("生成样例:")
    print(generated)


def parse_args() -> argparse.Namespace:
    """解析命令行参数，默认值保持小规模、容易跑通。"""

    parser = argparse.ArgumentParser(description="Pretrain a tiny GPT model.")
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--eval_batches", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--prompt", type=str, default="Learning")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
