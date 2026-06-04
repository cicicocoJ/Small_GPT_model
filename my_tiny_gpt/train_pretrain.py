r"""TinyStories-only MiniGPT pretraining script.

Dependency for Hugging Face datasets:
    pip install datasets
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tiny_tokenizer import GPT2Tokenizer, SimpleRegexTokenizer


class PretrainDataset:
    """Next-token prediction dataset built from one token sequence."""

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


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def safe_exp(value: float) -> float:
    if math.isnan(value):
        return float("nan")
    return math.exp(min(value, 50.0))


def resolve_project_path(script_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = script_dir / path
    return path


def make_output_dir(script_dir: Path, args: argparse.Namespace) -> Path:
    output_root = resolve_project_path(script_dir, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = timestamp if not args.run_name else f"{timestamp}_{args.run_name}"
    output_dir = output_root / folder_name
    suffix = 1
    while output_dir.exists():
        output_dir = output_root / f"{folder_name}_{suffix:02d}"
        suffix += 1
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def setup_hf_cache(script_dir: Path, cache_dir_value: str) -> Path:
    cache_dir = resolve_project_path(script_dir, cache_dir_value)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    (cache_dir / "hub").mkdir(parents=True, exist_ok=True)
    (cache_dir / "datasets").mkdir(parents=True, exist_ok=True)
    return cache_dir


def extract_texts(dataset, text_field: str) -> list[str]:
    texts = []
    for item in dataset:
        if text_field not in item:
            raise KeyError(f"text field '{text_field}' not found; available fields: {list(item.keys())}")
        text = str(item[text_field]).strip()
        if text:
            texts.append(text)
    return texts


def join_stories(texts: list[str]) -> str:
    return "\n<|endoftext|>\n".join(texts).strip()


def build_selected_tokenizer(tokenizer_name: str, train_text: str):
    if tokenizer_name == "regex":
        return SimpleRegexTokenizer.build_from_text(train_text)
    if tokenizer_name == "gpt2":
        return GPT2Tokenizer.build_from_text(train_text)
    raise ValueError(f"unsupported tokenizer: {tokenizer_name}")


def load_tinystories(script_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    cache_dir = setup_hf_cache(script_dir, args.cache_dir)
    disk_train_path = resolve_project_path(script_dir, args.hf_disk_train_path)
    disk_val_path = resolve_project_path(script_dir, args.hf_disk_val_path)
    saved_hf_to_disk = False

    # Import datasets only after Hugging Face cache variables are set.
    if args.hf_load_mode == "hub":
        from datasets import load_dataset

        ds_train = load_dataset(args.hf_dataset, split=args.hf_train_split, cache_dir=str(cache_dir))
        ds_val = load_dataset(args.hf_dataset, split=args.hf_val_split, cache_dir=str(cache_dir))
        if args.save_hf_to_disk:
            disk_train_path.parent.mkdir(parents=True, exist_ok=True)
            disk_val_path.parent.mkdir(parents=True, exist_ok=True)
            ds_train.save_to_disk(str(disk_train_path))
            ds_val.save_to_disk(str(disk_val_path))
            saved_hf_to_disk = True
    elif args.hf_load_mode == "disk":
        from datasets import load_from_disk

        ds_train = load_from_disk(str(disk_train_path))
        ds_val = load_from_disk(str(disk_val_path))
    else:
        raise ValueError(f"unsupported hf_load_mode: {args.hf_load_mode}")

    train_text = join_stories(extract_texts(ds_train, args.text_field))
    val_text = join_stories(extract_texts(ds_val, args.text_field))
    if args.clean_text:
        train_text = clean_text(train_text)
        val_text = clean_text(val_text)

    tokenizer = build_selected_tokenizer(args.tokenizer, train_text)
    train_token_ids = tokenizer.encode(train_text)
    val_token_ids = tokenizer.encode(val_text)
    train_dataset = PretrainDataset(train_token_ids, args.context_length, args.stride)
    val_dataset = PretrainDataset(val_token_ids, args.context_length, args.stride)

    return {
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_text": train_text,
        "val_text": val_text,
        "train_tokens": len(train_token_ids),
        "val_tokens": len(val_token_ids),
        "cache_dir": str(cache_dir),
        "hf_disk_train_path": str(disk_train_path),
        "hf_disk_val_path": str(disk_val_path),
        "saved_hf_to_disk": saved_hf_to_disk,
    }


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
    if penalty <= 1.0:
        return logits
    for token_id in set(generated_ids):
        value = logits[:, token_id]
        logits[:, token_id] = torch.where(value < 0, value * penalty, value / penalty)
    return logits


def banned_ngram_tokens(generated_ids: list[int], ngram_size: int) -> set[int]:
    if ngram_size <= 0 or len(generated_ids) < ngram_size - 1:
        return set()
    prefix = tuple(generated_ids[-(ngram_size - 1):]) if ngram_size > 1 else tuple()
    banned = set()
    for i in range(len(generated_ids) - ngram_size + 1):
        ngram = tuple(generated_ids[i : i + ngram_size])
        if ngram_size == 1 or ngram[:-1] == prefix:
            banned.add(ngram[-1])
    return banned


def sample_next_token(logits: torch.Tensor, generated_ids: list[int], args: argparse.Namespace) -> torch.Tensor:
    logits = logits.clone()
    logits = apply_repetition_penalty(logits, generated_ids, args.repetition_penalty)
    banned = banned_ngram_tokens(generated_ids, args.no_repeat_ngram_size)
    if banned:
        logits[:, list(banned)] = float("-inf")

    if not args.sample or args.temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / args.temperature
    if args.top_k > 0:
        top_k = min(args.top_k, logits.size(-1))
        top_values, _ = torch.topk(logits, top_k)
        threshold = top_values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def decode_for_sample(tokenizer, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except TypeError:
        text = tokenizer.decode(token_ids)
        return text.replace("<|endoftext|>", "")


def generate_text(model: MiniGPT, tokenizer, args: argparse.Namespace, device: torch.device) -> str:
    model.eval()
    ids = tokenizer.encode(args.prompt)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    eos_id = getattr(tokenizer, "eos_id", None)

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            input_cond = input_ids[:, -model.cfg.context_length:]
            logits = model(input_cond)[:, -1, :]
            generated_ids = input_ids.squeeze(0).tolist()
            next_id = sample_next_token(logits, generated_ids, args)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if eos_id is not None and int(next_id.item()) == int(eos_id):
                break

    text = decode_for_sample(tokenizer, input_ids.squeeze(0).tolist())
    if args.collapse_blank_lines:
        text = collapse_blank_lines(text)
    return text


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
    plt.title("MiniGPT Pretraining Loss")
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
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "vocab_path": str(vocab_path),
        "hf_dataset": args.hf_dataset,
        "hf_train_split": args.hf_train_split,
        "hf_val_split": args.hf_val_split,
        "hf_load_mode": args.hf_load_mode,
        "tokenizer_type": args.tokenizer,
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


def build_metrics(
    args: argparse.Namespace,
    cfg: MiniGPTConfig,
    data: dict[str, Any],
    tokenizer,
    parameters: int,
    best_val_loss: float,
    best_epoch: int | None,
    best_step: int | None,
    final_train_loss: float,
    final_val_loss: float,
    elapsed: float,
    output_dir: Path,
    best_model_path: Path,
    last_model_path: Path,
    sample_path: Path,
) -> dict[str, Any]:
    return {
        "hf_dataset": args.hf_dataset,
        "hf_train_split": args.hf_train_split,
        "hf_val_split": args.hf_val_split,
        "hf_load_mode": args.hf_load_mode,
        "cache_dir": data["cache_dir"],
        "hf_disk_train_path": data["hf_disk_train_path"],
        "hf_disk_val_path": data["hf_disk_val_path"],
        "saved_hf_to_disk": data["saved_hf_to_disk"],
        "tokenizer": args.tokenizer,
        "vocab_size": tokenizer.vocab_size,
        "train_text_chars": len(data["train_text"]),
        "val_text_chars": len(data["val_text"]),
        "train_tokens": data["train_tokens"],
        "val_tokens": data["val_tokens"],
        "train_samples": len(data["train_dataset"]),
        "val_samples": len(data["val_dataset"]),
        "parameters": parameters,
        "context_length": args.context_length,
        "stride": args.stride,
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "elapsed_seconds": elapsed,
        "output_dir": str(output_dir),
        "model_best_path": str(best_model_path) if args.save_best else "",
        "model_last_path": str(last_model_path),
        "sample_path": str(sample_path),
        "config": cfg.__dict__,
    }


def train(args: argparse.Namespace) -> None:
    global torch, F, DataLoader, MiniGPT, MiniGPTConfig, count_parameters

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from mini_gpt import MiniGPT, MiniGPTConfig, count_parameters

    script_dir = Path(__file__).resolve().parent
    output_dir = make_output_dir(script_dir, args)
    vocab_path = output_dir / "vocab.json"
    best_model_path = output_dir / "pretrain_model_best.pt"
    last_model_path = output_dir / "pretrain_model_last.pt"
    loss_csv_path = output_dir / "pretrain_loss.csv"
    loss_png_path = output_dir / "pretrain_loss.png"
    log_path = output_dir / "pretrain_log.txt"
    sample_path = output_dir / "pretrain_sample.txt"
    metrics_path = output_dir / "metrics.json"
    args_path = output_dir / "args.json"

    args_path.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    data = load_tinystories(script_dir, args)
    tokenizer = data["tokenizer"]
    train_dataset = data["train_dataset"]
    val_dataset = data["val_dataset"]
    tokenizer.save(vocab_path)

    drop_last = len(train_dataset) >= args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=drop_last)
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
    parameters = count_parameters(model)

    best_val_loss = float("inf")
    best_epoch = None
    best_step = None
    bad_eval_count = 0
    early_stopped = False
    global_step = 0
    loss_rows: list[dict[str, float]] = []
    start_time = time.time()

    log_lines = [
        "MiniGPT TinyStories pretraining log",
        f"hf_dataset: {args.hf_dataset}",
        f"hf_train_split: {args.hf_train_split}",
        f"hf_val_split: {args.hf_val_split}",
        f"hf_load_mode: {args.hf_load_mode}",
        f"cache_dir: {data['cache_dir']}",
        f"hf_disk_train_path: {data['hf_disk_train_path']}",
        f"hf_disk_val_path: {data['hf_disk_val_path']}",
        f"saved_hf_to_disk: {data['saved_hf_to_disk']}",
        f"tokenizer: {args.tokenizer}",
        f"vocab_size: {tokenizer.vocab_size}",
        f"train text chars: {len(data['train_text'])}",
        f"val text chars: {len(data['val_text'])}",
        f"train tokens: {data['train_tokens']}",
        f"val tokens: {data['val_tokens']}",
        f"train samples: {len(train_dataset)}",
        f"val samples: {len(val_dataset)}",
        f"model parameters: {parameters}",
        f"config: {json.dumps(cfg.__dict__, ensure_ascii=False)}",
        f"output_dir: {output_dir}",
        f"device: {device}",
        f"save_best: {args.save_best}",
        f"early_stopping_patience: {args.early_stopping_patience}",
        f"min_delta: {args.min_delta}",
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
            train_ppl = safe_exp(train_loss)
            val_ppl = safe_exp(val_loss)
            loss_rows.append({
                "epoch": epoch,
                "step": global_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_perplexity": train_ppl,
                "val_perplexity": val_ppl,
            })

            improved = val_loss < best_val_loss - args.min_delta
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch
                best_step = global_step
                bad_eval_count = 0
                if args.save_best:
                    torch.save(
                        make_checkpoint(model, cfg, args, vocab_path, train_loss, val_loss, best_val_loss, best_epoch, best_step, epoch, global_step),
                        best_model_path,
                    )
            else:
                bad_eval_count += 1

            line = (
                f"epoch {epoch:02d} step {global_step:06d} | "
                f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                f"train_perplexity {train_ppl:.2f} | val_perplexity {val_ppl:.2f} | "
                f"best_val_loss {best_val_loss:.4f} | bad_eval_count {bad_eval_count}"
            )
            print(line)
            log_lines.append(line)

            if args.early_stopping_patience > 0 and bad_eval_count >= args.early_stopping_patience:
                early_stopped = True
                stop_line = f"early stopping triggered at epoch {epoch}, step {global_step}"
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
        if args.save_best:
            torch.save(
                make_checkpoint(model, cfg, args, vocab_path, final_train_loss, final_val_loss, best_val_loss, best_epoch, best_step, final_epoch, global_step),
                best_model_path,
            )

    torch.save(
        make_checkpoint(model, cfg, args, vocab_path, final_train_loss, final_val_loss, best_val_loss, best_epoch, best_step, final_epoch, global_step),
        last_model_path,
    )

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
    metrics = build_metrics(
        args, cfg, data, tokenizer, parameters,
        best_val_loss, best_epoch, best_step, final_train_loss, final_val_loss,
        elapsed, output_dir, best_model_path, last_model_path, sample_path,
    )
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    log_lines.extend([
        f"best_val_loss: {best_val_loss:.4f}",
        f"best_epoch: {best_epoch}",
        f"best_step: {best_step}",
        f"bad_eval_count: {bad_eval_count}",
        f"early_stopped: {early_stopped}",
        f"final_train_loss: {final_train_loss:.4f}",
        f"final_val_loss: {final_val_loss:.4f}",
        f"elapsed_seconds: {elapsed:.2f}",
        f"model_best_path: {best_model_path if args.save_best else 'not saved (--save_best disabled)'}",
        f"model_last_path: {last_model_path}",
        f"sample_path: {sample_path}",
        f"metrics_path: {metrics_path}",
        "generated sample:",
        generated,
    ])
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    print("Training complete.")
    print(f"Output dir: {output_dir}")
    print(f"Last model: {last_model_path}")
    if args.save_best:
        print(f"Best model: {best_model_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Sample: {sample_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain MiniGPT on TinyStories.")
    parser.add_argument("--hf_dataset", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--hf_train_split", type=str, default="train[:2000]")
    parser.add_argument("--hf_val_split", type=str, default="validation[:200]")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--cache_dir", type=str, default="data/hf_cache")
    parser.add_argument("--hf_load_mode", type=str, default="hub", choices=["hub", "disk"])
    parser.add_argument("--save_hf_to_disk", action="store_true")
    parser.add_argument("--hf_disk_train_path", type=str, default="data/TinyStories_train")
    parser.add_argument("--hf_disk_val_path", type=str, default="data/TinyStories_val")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--tokenizer", type=str, default="regex", choices=["regex", "gpt2"])
    parser.add_argument("--clean_text", action="store_true")
    parser.add_argument("--collapse_blank_lines", action="store_true")
    parser.add_argument("--save_best", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--context_length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--eval_freq", type=int, default=50)
    parser.add_argument("--eval_batches", type=int, default=100)
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
