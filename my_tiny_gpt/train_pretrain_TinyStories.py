# NOTE: This script is kept for reference only.
# Recommended unified entry point:
#   python train_pretrain.py
# train_pretrain.py is now TinyStories-only, so this old file is deprecated.
r"""Pretrain MiniGPT on Hugging Face TinyStories.

Dependency:
    pip install datasets

Example:
    python train_pretrain_TinyStories.py --tokenizer regex --cpu
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from mini_gpt import MiniGPT, MiniGPTConfig, count_parameters
from tiny_tokenizer import GPT2Tokenizer, SimpleRegexTokenizer


class PretrainDataset(Dataset):
    """Next-token prediction dataset."""

    def __init__(self, token_ids: list[int], context_length: int, stride: int):
        if len(token_ids) <= context_length:
            raise ValueError("token count must be larger than context_length")
        self.input_ids = []
        self.target_ids = []
        for start in range(0, len(token_ids) - context_length, stride):
            x = token_ids[start : start + context_length]
            y = token_ids[start + 1 : start + context_length + 1]
            self.input_ids.append(torch.tensor(x, dtype=torch.long))
            self.target_ids.append(torch.tensor(y, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.target_ids[index]


def safe_exp(value: float) -> float:
    if math.isnan(value):
        return float("nan")
    return math.exp(min(value, 50.0))


def make_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_texts(dataset, text_field: str, max_examples: int | None = None) -> list[str]:
    """Extract non-empty stories from a Hugging Face dataset split."""

    texts = []
    count = len(dataset) if max_examples is None else min(len(dataset), max_examples)
    for i in range(count):
        item = dataset[i]
        if text_field not in item:
            raise KeyError(f"text field '{text_field}' not found; available fields: {list(item.keys())}")
        text = str(item[text_field]).strip()
        if text:
            texts.append(text)
    return texts


def join_stories(texts: list[str]) -> str:
    return "\n<|endoftext|>\n".join(texts).strip()


def load_tinystories_texts(args: argparse.Namespace) -> tuple[str, str, str]:
    """Load train/validation text from TinyStories with validation fallback."""

    ds_train = load_dataset(args.hf_dataset, split=args.hf_train_split)
    used_val_split = args.hf_val_split

    try:
        ds_val = load_dataset(args.hf_dataset, split=args.hf_val_split)
        train_texts = extract_texts(ds_train, args.text_field, args.max_train_examples)
        val_texts = extract_texts(ds_val, args.text_field, args.max_val_examples)
    except Exception as exc:
        print(f"Validation split load failed: {exc}")
        print("Falling back to a held-out tail from the loaded train split.")
        all_train_texts = extract_texts(ds_train, args.text_field, args.max_train_examples)
        if len(all_train_texts) < 2:
            raise ValueError("not enough train examples to create fallback validation split")
        fallback_val_size = args.max_val_examples or max(1, min(500, len(all_train_texts) // 10))
        fallback_val_size = min(fallback_val_size, len(all_train_texts) - 1)
        train_texts = all_train_texts[:-fallback_val_size]
        val_texts = all_train_texts[-fallback_val_size:]
        used_val_split = f"fallback_from_{args.hf_train_split}[-{fallback_val_size}:]"

    return join_stories(train_texts), join_stories(val_texts), used_val_split


def build_selected_tokenizer(tokenizer_name: str, train_text: str):
    """Build tokenizer using only training text to avoid validation leakage."""

    if tokenizer_name == "regex":
        return SimpleRegexTokenizer.build_from_text(train_text)
    if tokenizer_name == "gpt2":
        return GPT2Tokenizer.build_from_text(train_text)
    raise ValueError(f"unsupported tokenizer: {tokenizer_name}")


def calc_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor, model: MiniGPT, device: torch.device) -> torch.Tensor:
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_batch.reshape(-1))


def evaluate_loss(model: MiniGPT, data_loader: DataLoader, device: torch.device, max_batches: int) -> float:
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


def apply_repetition_penalty(logits: torch.Tensor, generated_ids: list[int], penalty: float) -> torch.Tensor:
    """Apply a simple repetition penalty to previously generated tokens."""

    if penalty <= 1.0:
        return logits
    for token_id in set(generated_ids):
        value = logits[:, token_id]
        logits[:, token_id] = torch.where(value < 0, value * penalty, value / penalty)
    return logits


def banned_ngram_tokens(generated_ids: list[int], ngram_size: int) -> set[int]:
    """Return tokens that would repeat an existing n-gram."""

    if ngram_size <= 0 or len(generated_ids) < ngram_size - 1:
        return set()
    prefix = tuple(generated_ids[-(ngram_size - 1):]) if ngram_size > 1 else tuple()
    banned = set()
    for i in range(len(generated_ids) - ngram_size + 1):
        ngram = tuple(generated_ids[i : i + ngram_size])
        if ngram_size == 1 or ngram[:-1] == prefix:
            banned.add(ngram[-1])
    return banned


def sample_next_token(
    logits: torch.Tensor,
    generated_ids: list[int],
    temperature: float,
    top_k: int,
    sample: bool,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> torch.Tensor:
    logits = logits.clone()
    logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    banned = banned_ngram_tokens(generated_ids, no_repeat_ngram_size)
    if banned:
        logits[:, list(banned)] = float("-inf")

    if not sample or temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        top_values, _ = torch.topk(logits, top_k)
        threshold = top_values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_text(model: MiniGPT, tokenizer, args: argparse.Namespace, device: torch.device) -> str:
    model.eval()
    ids = tokenizer.encode(args.prompt)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            input_cond = input_ids[:, -model.cfg.context_length:]
            logits = model(input_cond)[:, -1, :]
            generated_ids = input_ids.squeeze(0).tolist()
            next_id = sample_next_token(
                logits=logits,
                generated_ids=generated_ids,
                temperature=args.temperature,
                top_k=args.top_k,
                sample=args.sample,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
            input_ids = torch.cat([input_ids, next_id], dim=1)

    return tokenizer.decode(input_ids.squeeze(0).tolist())


def save_loss_csv(rows: list[dict[str, float]], path: Path) -> None:
    fieldnames = ["epoch", "step", "train_loss", "val_loss", "train_perplexity", "val_perplexity"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_loss_plot(rows: list[dict[str, float]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skip saving loss plot.")
        return
    steps = [row["step"] for row in rows]
    train_losses = [row["train_loss"] for row in rows]
    val_losses = [row["val_loss"] for row in rows]
    plt.figure(figsize=(6, 4))
    plt.plot(steps, train_losses, label="train loss")
    plt.plot(steps, val_losses, label="val loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("MiniGPT TinyStories Pretraining Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def make_checkpoint(
    model: MiniGPT,
    cfg: MiniGPTConfig,
    args: argparse.Namespace,
    vocab_path: Path,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    best_epoch: int | None,
    best_step: int | None,
    epoch: int,
    global_step: int,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "vocab_path": str(vocab_path),
        "tokenizer_type": args.tokenizer,
        "hf_dataset": args.hf_dataset,
        "hf_train_split": args.hf_train_split,
        "hf_val_split": args.hf_val_split,
        "epoch": epoch,
        "global_step": global_step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_perplexity": safe_exp(train_loss),
        "val_perplexity": safe_exp(val_loss),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_step": best_step,
    }


def train(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    output_arg = Path(args.output_dir)
    if not output_arg.is_absolute():
        output_arg = script_dir / output_arg
    output_dir = make_output_dir(output_arg)
    vocab_path = output_dir / "vocab.json"
    best_model_path = output_dir / "pretrain_model_best.pt"
    last_model_path = output_dir / "pretrain_model_last.pt"
    loss_csv_path = output_dir / "pretrain_loss.csv"
    loss_png_path = output_dir / "pretrain_loss.png"
    log_path = output_dir / "pretrain_log.txt"
    sample_path = output_dir / "pretrain_sample.txt"

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_text, val_text, used_val_split = load_tinystories_texts(args)
    tokenizer = build_selected_tokenizer(args.tokenizer, train_text)
    tokenizer.save(vocab_path)

    train_token_ids = tokenizer.encode(train_text)
    val_token_ids = tokenizer.encode(val_text)
    train_dataset = PretrainDataset(train_token_ids, args.context_length, args.stride)
    val_dataset = PretrainDataset(val_token_ids, args.context_length, args.stride)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

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

    best_val_loss = float("inf")
    best_epoch = None
    best_step = None
    bad_eval_count = 0
    early_stopped = False
    global_step = 0
    loss_rows = []
    start_time = time.time()

    log_lines = [
        "MiniGPT TinyStories pretraining log",
        f"dataset name: {args.hf_dataset}",
        f"train split: {args.hf_train_split}",
        f"val split: {used_val_split}",
        f"tokenizer type: {args.tokenizer}",
        f"device: {device}",
        f"train text chars: {len(train_text)}",
        f"val text chars: {len(val_text)}",
        f"train tokens: {len(train_token_ids)}",
        f"val tokens: {len(val_token_ids)}",
        f"vocab_size: {tokenizer.vocab_size}",
        f"train samples: {len(train_dataset)}",
        f"val samples: {len(val_dataset)}",
        f"model parameters: {count_parameters(model)}",
        f"config: {json.dumps(cfg.__dict__, ensure_ascii=False)}",
    ]
    print("\n".join(log_lines))

    final_epoch = 0
    for epoch in range(1, args.epochs + 1):
        final_epoch = epoch
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % args.eval_freq != 0:
                continue

            train_loss = evaluate_loss(model, train_loader, device, args.eval_batches)
            val_loss = evaluate_loss(model, val_loader, device, args.eval_batches)
            row = {
                "epoch": epoch,
                "step": global_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_perplexity": safe_exp(train_loss),
                "val_perplexity": safe_exp(val_loss),
            }
            loss_rows.append(row)

            improved = val_loss < best_val_loss - args.min_delta
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch
                best_step = global_step
                bad_eval_count = 0
                torch.save(make_checkpoint(model, cfg, args, vocab_path, train_loss, val_loss, best_val_loss, best_epoch, best_step, epoch, global_step), best_model_path)
            else:
                bad_eval_count += 1

            line = (
                f"epoch {epoch:02d} step {global_step:06d} | "
                f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                f"train_perplexity {row['train_perplexity']:.2f} | val_perplexity {row['val_perplexity']:.2f} | "
                f"best_val_loss {best_val_loss:.4f} | bad_eval_count {bad_eval_count}"
            )
            print(line)
            log_lines.append(line)

            if args.early_stopping_patience > 0 and bad_eval_count >= args.early_stopping_patience:
                early_stopped = True
                stop_line = f"early stopping at epoch {epoch}, step {global_step}"
                print(stop_line)
                log_lines.append(stop_line)
                break
        if early_stopped:
            break

    final_train_loss = evaluate_loss(model, train_loader, device, args.eval_batches)
    final_val_loss = evaluate_loss(model, val_loader, device, args.eval_batches)
    if math.isinf(best_val_loss):
        best_val_loss = final_val_loss
        best_epoch = final_epoch
        best_step = global_step
        torch.save(make_checkpoint(model, cfg, args, vocab_path, final_train_loss, final_val_loss, best_val_loss, best_epoch, best_step, final_epoch, global_step), best_model_path)

    torch.save(make_checkpoint(model, cfg, args, vocab_path, final_train_loss, final_val_loss, best_val_loss, best_epoch, best_step, final_epoch, global_step), last_model_path)

    loss_rows.append({
        "epoch": final_epoch,
        "step": global_step,
        "train_loss": final_train_loss,
        "val_loss": final_val_loss,
        "train_perplexity": safe_exp(final_train_loss),
        "val_perplexity": safe_exp(final_val_loss),
    })
    save_loss_csv(loss_rows, loss_csv_path)
    save_loss_plot(loss_rows, loss_png_path)

    generated = generate_text(model, tokenizer, args, device)
    sample_path.write_text(generated, encoding="utf-8")

    elapsed = time.time() - start_time
    log_lines.extend([
        f"best_val_loss: {best_val_loss:.4f}",
        f"best_epoch: {best_epoch}",
        f"best_step: {best_step}",
        f"final_train_loss: {final_train_loss:.4f}",
        f"final_val_loss: {final_val_loss:.4f}",
        f"elapsed_seconds: {elapsed:.2f}",
        f"best_model_path: {best_model_path}",
        f"last_model_path: {last_model_path}",
        f"loss_csv_path: {loss_csv_path}",
        f"sample_path: {sample_path}",
        "generated sample:",
        generated,
    ])
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    print("Training complete.")
    print(f"Best model: {best_model_path}")
    print(f"Last model: {last_model_path}")
    print(f"Sample: {sample_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain MiniGPT on TinyStories.")
    parser.add_argument("--hf_dataset", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--hf_train_split", type=str, default="train[:5000]")
    parser.add_argument("--hf_val_split", type=str, default="validation[:500]")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_val_examples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs_tinystories")
    parser.add_argument("--tokenizer", choices=["regex", "gpt2"], default="regex")
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eval_freq", type=int, default=50)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
