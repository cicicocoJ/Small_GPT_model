r"""Fine-tune MiniGPT for Story Ending Polarity Classification.

Windows VS Code example:
    cd H:\学校\人工智能基础\Small_gpt_model\my_tiny_gpt
    conda activate minigpt
    python finetune_classifier.py --cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mini_gpt import MiniGPT, MiniGPTConfig
from tiny_tokenizer import load_tokenizer


LABEL_TO_ID = {"negative": 0, "positive": 1}
ID_TO_LABEL = {0: "negative", 1: "positive"}


class StoryEndingDataset(Dataset):
    """将故事文本编码成 input_ids、attention_mask 和分类标签。"""

    def __init__(self, rows: list[dict[str, str]], tokenizer, max_length: int):
        self.rows = rows
        self.samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        pad_id = int(getattr(tokenizer, "pad_id", getattr(tokenizer, "eos_id", 0)))

        for row in rows:
            text = row["text"].strip()
            label_name = row["label"].strip().lower()
            if label_name not in LABEL_TO_ID:
                raise ValueError(f"Unsupported label '{row['label']}'. Use positive or negative.")

            token_ids = tokenizer.encode(text)[:max_length]
            if not token_ids:
                token_ids = [pad_id]

            attention_mask = [1] * len(token_ids)
            padding = max_length - len(token_ids)
            token_ids = token_ids + [pad_id] * padding
            attention_mask = attention_mask + [0] * padding

            self.samples.append(
                (
                    torch.tensor(token_ids, dtype=torch.long),
                    torch.tensor(attention_mask, dtype=torch.long),
                    torch.tensor(LABEL_TO_ID[label_name], dtype=torch.long),
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.samples[index]


class MiniGPTClassifier(nn.Module):
    """不修改 MiniGPT 源码，只在预训练主体后增加线性分类头。"""

    def __init__(self, cfg: MiniGPTConfig, num_labels: int = 2, pooling: str = "last"):
        super().__init__()
        self.gpt = MiniGPT(cfg)
        self.pooling = pooling
        self.classifier = nn.Linear(cfg.emb_dim, num_labels)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """复用 MiniGPT 的内部模块，取 hidden states 而不是 vocab logits。"""

        _, seq_len = input_ids.shape
        if seq_len > self.gpt.cfg.context_length:
            raise ValueError(f"input length {seq_len} exceeds context_length={self.gpt.cfg.context_length}")

        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.gpt.tok_emb(input_ids) + self.gpt.pos_emb(positions)
        x = self.gpt.drop_emb(x)
        x = self.gpt.trf_blocks(x)
        return self.gpt.final_norm(x)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(input_ids)

        if self.pooling == "last":
            # 根据 attention_mask 找最后一个真实 token，避免取到 padding 位置。
            last_indices = attention_mask.sum(dim=1).clamp(min=1) - 1
            batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
            pooled = hidden[batch_indices, last_indices]
        elif self.pooling == "mean":
            # 平均池化时只统计真实 token。
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")

        return self.classifier(pooled)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_project_path(script_dir: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = script_dir / path
    return path


def make_output_dir(script_dir: Path, output_root: str, run_name: str) -> Path:
    root = resolve_project_path(script_dir, output_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = timestamp if not run_name else f"{timestamp}_{run_name}"
    output_dir = root / folder_name
    suffix = 1
    while output_dir.exists():
        output_dir = root / f"{folder_name}_{suffix:02d}"
        suffix += 1
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def find_latest_pretrain_checkpoint(script_dir: Path, output_root: str) -> Path:
    root = resolve_project_path(script_dir, output_root)
    if not root.exists():
        raise FileNotFoundError(f"Cannot find output root: {root}")

    best_candidates = sorted(root.glob("*/pretrain_model_best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if best_candidates:
        return best_candidates[0]

    last_candidates = sorted(root.glob("*/pretrain_model_last.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if last_candidates:
        return last_candidates[0]

    raise FileNotFoundError(f"No pretrain_model_best.pt or pretrain_model_last.pt found under {root}")


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_classifier_checkpoint(model: MiniGPTClassifier, checkpoint_path: Path, device: torch.device) -> int:
    checkpoint = torch_load(checkpoint_path, device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Classifier checkpoint must contain model_state_dict: {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    return int(checkpoint.get("epoch", 0))


def get_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise ValueError("Checkpoint must be a state_dict or contain 'model_state_dict'.")


def recover_config(checkpoint: Any, checkpoint_path: Path, tokenizer) -> MiniGPTConfig:
    cfg_data: dict[str, Any] = {}
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        cfg_data.update(checkpoint["config"])

    metrics = load_json_if_exists(checkpoint_path.parent / "metrics.json")
    args = load_json_if_exists(checkpoint_path.parent / "args.json")
    if not cfg_data and isinstance(metrics.get("config"), dict):
        cfg_data.update(metrics["config"])
    if not cfg_data:
        for key in ["vocab_size", "context_length", "emb_dim", "n_heads", "n_layers", "dropout", "qkv_bias"]:
            if key in metrics:
                cfg_data[key] = metrics[key]
            elif key in args:
                cfg_data[key] = args[key]

    required = ["vocab_size", "context_length", "emb_dim", "n_heads", "n_layers", "dropout"]
    missing = [key for key in required if key not in cfg_data]
    if missing:
        raise ValueError(f"Cannot recover MiniGPTConfig. Missing fields: {missing}")

    cfg_data.setdefault("qkv_bias", False)
    cfg_data["vocab_size"] = int(cfg_data["vocab_size"])
    if cfg_data["vocab_size"] != tokenizer.vocab_size:
        raise ValueError(
            f"Checkpoint vocab_size={cfg_data['vocab_size']} does not match tokenizer vocab_size={tokenizer.vocab_size}. "
            "Please use the vocab.json saved with this checkpoint."
        )

    return MiniGPTConfig(**cfg_data)


def read_finetune_rows(data_path: Path) -> list[dict[str, str]]:
    with data_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "text" not in rows[0] or "label" not in rows[0]:
        raise ValueError("finetune.csv must contain text,label columns.")
    for row in rows:
        row["label"] = row["label"].strip().lower()
        if row["label"] not in LABEL_TO_ID:
            raise ValueError(f"Invalid label '{row['label']}'. Only positive/negative are allowed.")
    return rows


def stratified_split(
    rows: list[dict[str, str]],
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, str]]] = {label: [] for label in LABEL_TO_ID}
    for row in rows:
        by_label[row["label"]].append(row)

    if any(len(items) < 3 for items in by_label.values()):
        raise ValueError("Each label needs at least 3 samples for train/val/test split.")

    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []
    for label, items in by_label.items():
        rng.shuffle(items)
        n = len(items)
        train_size = max(1, round(n * train_ratio))
        val_size = max(1, round(n * val_ratio))
        if train_size + val_size >= n:
            val_size = 1
            train_size = n - 2
        train_rows.extend(items[:train_size])
        val_rows.extend(items[train_size : train_size + val_size])
        test_rows.extend(items[train_size + val_size :])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows


def create_dataloaders(
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    tokenizer,
    max_length: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = StoryEndingDataset(train_rows, tokenizer, max_length)
    val_ds = StoryEndingDataset(val_rows, tokenizer, max_length)
    test_ds = StoryEndingDataset(test_rows, tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def set_finetune_mode(model: MiniGPTClassifier, mode: str) -> None:
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    if mode == "head_only":
        return
    if mode == "last_block":
        for param in model.gpt.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.gpt.final_norm.parameters():
            param.requires_grad = True
        return
    if mode == "full":
        for param in model.parameters():
            param.requires_grad = True
        return
    raise ValueError(f"Unsupported finetune mode: {mode}")


def parameter_stats(model: nn.Module) -> tuple[int, int, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable / total if total else 0.0
    return total, trainable, ratio


def load_pretrained_classifier(
    checkpoint_path: Path,
    vocab_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[MiniGPTClassifier, MiniGPTConfig, Any]:
    tokenizer = load_tokenizer(vocab_path)
    checkpoint = torch_load(checkpoint_path, device)
    cfg = recover_config(checkpoint, checkpoint_path, tokenizer)
    if args.max_length > int(cfg.context_length):
        raise ValueError("--max_length cannot be larger than the pretraining context_length.")
    model = MiniGPTClassifier(cfg, num_labels=2, pooling=args.pooling)
    state_dict = get_state_dict(checkpoint)
    model.gpt.load_state_dict(state_dict)
    model.to(device)
    set_finetune_mode(model, args.finetune_mode)
    return model, cfg, tokenizer


def run_one_epoch(model: MiniGPTClassifier, loader: DataLoader, optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=-1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def evaluate(model: MiniGPTClassifier, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)
            preds = torch.argmax(logits, dim=-1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def is_better_validation(
    val_acc: float,
    val_loss: float,
    best_val_acc: float,
    best_val_loss: float,
    min_delta: float,
) -> bool:
    if val_acc > best_val_acc + min_delta:
        return True
    if abs(val_acc - best_val_acc) <= min_delta and val_loss < best_val_loss - min_delta:
        return True
    return False


def make_checkpoint(model: MiniGPTClassifier, cfg: MiniGPTConfig, args: argparse.Namespace, epoch: int) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "gpt_state_dict": model.gpt.state_dict(),
        "classifier_state_dict": model.classifier.state_dict(),
        "config": cfg.__dict__,
        "task": "story_ending_polarity_classification",
        "label_mapping": LABEL_TO_ID,
        "pooling": args.pooling,
        "finetune_mode": args.finetune_mode,
        "epoch": epoch,
    }


def save_history_csv(history: list[dict[str, float]], path: Path) -> None:
    fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_predictions(
    model: MiniGPTClassifier,
    rows: list[dict[str, str]],
    tokenizer,
    max_length: int,
    device: torch.device,
    path: Path,
) -> list[dict[str, Any]]:
    dataset = StoryEndingDataset(rows, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    results: list[dict[str, Any]] = []

    model.eval()
    offset = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1).cpu()
            preds = torch.argmax(probs, dim=-1)

            for i in range(labels.size(0)):
                row = rows[offset + i]
                true_id = int(labels[i].item())
                pred_id = int(preds[i].item())
                results.append(
                    {
                        "text": row["text"],
                        "true_label": ID_TO_LABEL[true_id],
                        "pred_label": ID_TO_LABEL[pred_id],
                        "prob_negative": float(probs[i, 0].item()),
                        "prob_positive": float(probs[i, 1].item()),
                        "correct": pred_id == true_id,
                    }
                )
            offset += labels.size(0)

    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["text", "true_label", "pred_label", "prob_negative", "prob_positive", "correct"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    return results


def compute_classification_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    tn = fp = fn = tp = 0
    for item in predictions:
        true_id = LABEL_TO_ID[item["true_label"]]
        pred_id = LABEL_TO_ID[item["pred_label"]]
        if true_id == 0 and pred_id == 0:
            tn += 1
        elif true_id == 0 and pred_id == 1:
            fp += 1
        elif true_id == 1 and pred_id == 0:
            fn += 1
        elif true_id == 1 and pred_id == 1:
            tp += 1

    def safe_div(num: float, den: float) -> float:
        return num / den if den else 0.0

    precision_negative = safe_div(tn, tn + fn)
    recall_negative = safe_div(tn, tn + fp)
    f1_negative = safe_div(2 * precision_negative * recall_negative, precision_negative + recall_negative)
    precision_positive = safe_div(tp, tp + fp)
    recall_positive = safe_div(tp, tp + fn)
    f1_positive = safe_div(2 * precision_positive * recall_positive, precision_positive + recall_positive)

    negative_support = tn + fp
    positive_support = tp + fn
    total_support = negative_support + positive_support
    macro_precision = (precision_negative + precision_positive) / 2
    macro_recall = (recall_negative + recall_positive) / 2
    macro_f1 = (f1_negative + f1_positive) / 2
    weighted_f1 = safe_div(
        f1_negative * negative_support + f1_positive * positive_support,
        total_support,
    )

    return {
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "precision_negative": precision_negative,
        "recall_negative": recall_negative,
        "f1_negative": f1_negative,
        "precision_positive": precision_positive,
        "recall_positive": recall_positive,
        "f1_positive": f1_positive,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "negative_support": negative_support,
        "positive_support": positive_support,
    }


def save_errors(predictions: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    errors = []
    for item in predictions:
        if item["correct"]:
            continue
        error_type = (
            "negative_to_positive"
            if item["true_label"] == "negative" and item["pred_label"] == "positive"
            else "positive_to_negative"
        )
        errors.append(
            {
                "text": item["text"],
                "true_label": item["true_label"],
                "pred_label": item["pred_label"],
                "prob_negative": item["prob_negative"],
                "prob_positive": item["prob_positive"],
                "error_type": error_type,
            }
        )

    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["text", "true_label", "pred_label", "prob_negative", "prob_positive", "error_type"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)
    return errors


def save_classification_report(
    path: Path,
    args: argparse.Namespace,
    data_path: Path,
    checkpoint_path: Path,
    train_size: int,
    val_size: int,
    test_size: int,
    best_epoch: int,
    best_val_loss: float,
    best_val_acc: float,
    test_loss: float,
    test_acc: float,
    test_metrics: dict[str, Any],
    errors: list[dict[str, Any]],
) -> None:
    cm = test_metrics["confusion_matrix"]
    negative_to_positive = sum(1 for item in errors if item["error_type"] == "negative_to_positive")
    positive_to_negative = sum(1 for item in errors if item["error_type"] == "positive_to_negative")
    lines = [
        "Classification Report",
        "",
        "Task: story_ending_polarity_classification",
        f"Data path: {data_path}",
        f"Checkpoint path: {checkpoint_path}",
        f"Finetune mode: {args.finetune_mode}",
        f"Pooling: {args.pooling}",
        f"Train/val/test size: {train_size}/{val_size}/{test_size}",
        f"Best epoch: {best_epoch}",
        f"Best val loss: {best_val_loss:.6f}",
        f"Best val acc: {best_val_acc:.6f}",
        f"Test loss: {test_loss:.6f}",
        f"Test acc: {test_acc:.6f}",
        "",
        "Confusion matrix:",
        f"tn={cm['tn']} fp={cm['fp']} fn={cm['fn']} tp={cm['tp']}",
        "",
        "Per-class metrics:",
        (
            "negative: "
            f"precision={test_metrics['precision_negative']:.6f} "
            f"recall={test_metrics['recall_negative']:.6f} "
            f"f1={test_metrics['f1_negative']:.6f}"
        ),
        (
            "positive: "
            f"precision={test_metrics['precision_positive']:.6f} "
            f"recall={test_metrics['recall_positive']:.6f} "
            f"f1={test_metrics['f1_positive']:.6f}"
        ),
        f"Macro precision: {test_metrics['macro_precision']:.6f}",
        f"Macro recall: {test_metrics['macro_recall']:.6f}",
        f"Macro F1: {test_metrics['macro_f1']:.6f}",
        f"Weighted F1: {test_metrics['weighted_f1']:.6f}",
        "",
        "Main error patterns:",
        f"negative_to_positive: {negative_to_positive}",
        f"positive_to_negative: {positive_to_negative}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def save_sample(predictions: list[dict[str, Any]], path: Path, max_items: int = 12) -> None:
    lines = ["Story Ending Polarity Classification samples", "labels: negative=0, positive=1", ""]
    for item in predictions[:max_items]:
        lines.append(
            f"true={item['true_label']} pred={item['pred_label']} "
            f"p_neg={item['prob_negative']:.3f} p_pos={item['prob_positive']:.3f} "
            f"correct={item['correct']} | {item['text']}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_curves(history: list[dict[str, float]], loss_path: Path, acc_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skip saving plots.")
        return

    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [row["train_loss"] for row in history], label="train loss")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Fine-tuning Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [row["train_acc"] for row in history], label="train accuracy")
    plt.plot(epochs, [row["val_acc"] for row in history], label="val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.title("Fine-tuning Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path, dpi=150)
    plt.close()


def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    output_dir = make_output_dir(script_dir, args.output_root, args.run_name)
    args_path = output_dir / "args.json"
    best_model_path = output_dir / "finetune_classifier_best.pt"
    last_model_path = output_dir / "finetune_classifier_last.pt"
    metrics_path = output_dir / "metrics.json"
    history_path = output_dir / "finetune_history.csv"
    loss_path = output_dir / "finetune_loss.png"
    acc_path = output_dir / "finetune_accuracy.png"
    predictions_path = output_dir / "finetune_predictions.csv"
    errors_path = output_dir / "finetune_errors.csv"
    report_path = output_dir / "classification_report.txt"
    sample_path = output_dir / "finetune_sample.txt"
    log_path = output_dir / "finetune_log.txt"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    checkpoint_path = (
        resolve_project_path(script_dir, args.pretrain_checkpoint)
        if args.pretrain_checkpoint
        else find_latest_pretrain_checkpoint(script_dir, args.output_root)
    )
    vocab_path = resolve_project_path(script_dir, args.vocab_path) if args.vocab_path else checkpoint_path.parent / "vocab.json"
    data_path = resolve_project_path(script_dir, args.data_path)

    if not vocab_path.exists():
        raise FileNotFoundError(f"Cannot find vocab.json: {vocab_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find finetune data: {data_path}")

    args_path.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    model, cfg, tokenizer = load_pretrained_classifier(checkpoint_path, vocab_path, args, device)
    max_length = args.max_length if args.max_length > 0 else cfg.context_length

    rows = read_finetune_rows(data_path)
    train_rows, val_rows, test_rows = stratified_split(rows, args.seed)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_rows, val_rows, test_rows, tokenizer, max_length, args.batch_size, args.num_workers
    )

    total_params, trainable_params, trainable_ratio = parameter_stats(model)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    log_lines = [
        "MiniGPT story ending polarity fine-tuning log",
        f"task: story_ending_polarity_classification",
        f"device: {device}",
        f"pretrain_checkpoint: {checkpoint_path}",
        f"vocab_path: {vocab_path}",
        f"data_path: {data_path}",
        f"output_dir: {output_dir}",
        f"finetune_mode: {args.finetune_mode}",
        f"pooling: {args.pooling}",
        f"max_length: {max_length}",
        f"total_parameters: {total_params}",
        f"trainable_parameters: {trainable_params}",
        f"trainable_ratio: {trainable_ratio:.6f}",
        f"train/val/test sizes: {len(train_rows)}/{len(val_rows)}/{len(test_rows)}",
        f"early_stopping_patience: {args.early_stopping_patience}",
        f"min_delta: {args.min_delta}",
        f"config: {json.dumps(cfg.__dict__, ensure_ascii=False)}",
    ]
    print("\n".join(log_lines))

    history: list[dict[str, float]] = []
    best_epoch = 0
    best_val_loss = float("inf")
    best_val_acc = -1.0
    bad_epoch_count = 0
    stopped_epoch = 0
    early_stopped = False
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)

        improved = is_better_validation(val_acc, val_loss, best_val_acc, best_val_loss, args.min_delta)
        if improved:
            best_epoch = epoch
            best_val_acc = val_acc
            best_val_loss = val_loss
            bad_epoch_count = 0
            torch.save(make_checkpoint(model, cfg, args, epoch), best_model_path)
        else:
            bad_epoch_count += 1

        line = (
            f"epoch {epoch:02d} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f} | bad_epoch_count {bad_epoch_count}"
        )
        print(line)
        log_lines.append(line)

        if args.early_stopping_patience > 0 and bad_epoch_count >= args.early_stopping_patience:
            stopped_epoch = epoch
            early_stopped = True
            stop_line = (
                f"early stopped at epoch {epoch}; best_epoch {best_epoch}; "
                f"best_val_acc {best_val_acc:.4f}; best_val_loss {best_val_loss:.4f}"
            )
            print(stop_line)
            log_lines.append(stop_line)
            break

    final_train_loss = history[-1]["train_loss"]
    final_train_acc = history[-1]["train_acc"]
    final_val_loss = history[-1]["val_loss"]
    final_val_acc = history[-1]["val_acc"]
    final_epoch = history[-1]["epoch"]
    torch.save(make_checkpoint(model, cfg, args, final_epoch), last_model_path)

    if not best_model_path.exists():
        torch.save(make_checkpoint(model, cfg, args, final_epoch), best_model_path)
        best_epoch = final_epoch
        best_val_loss = final_val_loss
        best_val_acc = final_val_acc

    loaded_best_epoch = load_classifier_checkpoint(model, best_model_path, device)
    test_line = f"Testing best checkpoint from epoch {loaded_best_epoch or best_epoch}"
    print(test_line)
    log_lines.append(test_line)
    test_loss, test_acc = evaluate(model, test_loader, device)

    save_history_csv(history, history_path)
    plot_curves(history, loss_path, acc_path)
    predictions = save_predictions(model, test_rows, tokenizer, max_length, device, predictions_path)
    test_metrics = compute_classification_metrics(predictions)
    errors = save_errors(predictions, errors_path)
    save_classification_report(
        report_path,
        args,
        data_path,
        checkpoint_path,
        len(train_rows),
        len(val_rows),
        len(test_rows),
        best_epoch,
        best_val_loss,
        best_val_acc,
        test_loss,
        test_acc,
        test_metrics,
        errors,
    )
    save_sample(predictions, sample_path)

    elapsed = time.time() - start_time
    metrics = {
        "task": "story_ending_polarity_classification",
        "num_labels": 2,
        "label_mapping": LABEL_TO_ID,
        "finetune_mode": args.finetune_mode,
        "pooling": args.pooling,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_ratio": trainable_ratio,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "test_size": len(test_rows),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "final_train_loss": final_train_loss,
        "final_train_acc": final_train_acc,
        "final_val_loss": final_val_loss,
        "final_val_acc": final_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "early_stopping_patience": args.early_stopping_patience,
        "min_delta": args.min_delta,
        "stopped_epoch": stopped_epoch,
        "early_stopped": early_stopped,
        "used_best_checkpoint": True,
        "bad_epoch_count": bad_epoch_count,
        "confusion_matrix": test_metrics["confusion_matrix"],
        "precision_negative": test_metrics["precision_negative"],
        "recall_negative": test_metrics["recall_negative"],
        "f1_negative": test_metrics["f1_negative"],
        "precision_positive": test_metrics["precision_positive"],
        "recall_positive": test_metrics["recall_positive"],
        "f1_positive": test_metrics["f1_positive"],
        "macro_precision": test_metrics["macro_precision"],
        "macro_recall": test_metrics["macro_recall"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
        "num_test_errors": len(errors),
        "pretrain_checkpoint": str(checkpoint_path),
        "vocab_path": str(vocab_path),
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "elapsed_seconds": elapsed,
        "max_length": max_length,
        "history": history,
        "config": cfg.__dict__,
    }
    save_metrics(metrics, metrics_path)

    log_lines.extend(
        [
            f"best_epoch: {best_epoch}",
            f"best_val_loss: {best_val_loss:.4f}",
            f"best_val_acc: {best_val_acc:.4f}",
            f"early_stopped: {early_stopped}",
            f"stopped_epoch: {stopped_epoch}",
            f"final_train_loss: {final_train_loss:.4f}",
            f"final_train_acc: {final_train_acc:.4f}",
            f"final_val_loss: {final_val_loss:.4f}",
            f"final_val_acc: {final_val_acc:.4f}",
            f"test_loss: {test_loss:.4f}",
            f"test_acc: {test_acc:.4f}",
            f"macro_f1: {test_metrics['macro_f1']:.4f}",
            f"weighted_f1: {test_metrics['weighted_f1']:.4f}",
            f"num_test_errors: {len(errors)}",
            f"elapsed_seconds: {elapsed:.2f}",
            f"best_model_path: {best_model_path}",
            f"last_model_path: {last_model_path}",
            f"metrics_path: {metrics_path}",
            f"predictions_path: {predictions_path}",
            f"errors_path: {errors_path}",
            f"classification_report_path: {report_path}",
            f"sample_path: {sample_path}",
        ]
    )
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    print("Fine-tuning complete.")
    print(f"Output dir: {output_dir}")
    print(f"Best model: {best_model_path}")
    print(f"Last model: {last_model_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Errors: {errors_path}")
    print(f"Classification report: {report_path}")
    print(f"Sample: {sample_path}")
    print(f"Test accuracy: {test_acc:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MiniGPT for story ending polarity classification.")
    parser.add_argument("--data_path", type=str, default="data/finetune.csv")
    parser.add_argument("--pretrain_checkpoint", type=str, default="")
    parser.add_argument("--vocab_path", type=str, default="")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default="finetune_story_ending")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=0, help="0 means use pretraining context_length")
    parser.add_argument("--pooling", type=str, default="mean", choices=["last", "mean"])
    parser.add_argument("--finetune_mode", type=str, default="full", choices=["last_block", "full", "head_only"])
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
