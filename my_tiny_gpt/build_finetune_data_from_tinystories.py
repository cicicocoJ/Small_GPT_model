r"""Build a manually labeled fine-tuning dataset from TinyStories.

Workflow:
    python build_finetune_data_from_tinystories.py extract --num_samples 300 --split train --start_index 5000
    # Open data/finetune_annotation_candidates.csv and fill label with positive/negative/discard.
    python build_finetune_data_from_tinystories.py make_csv --balance
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path
from typing import Any


VALID_LABELS = {"positive", "negative", "discard", ""}
FINAL_LABELS = {"positive", "negative"}
NEGATIVE_ENDING_KEYWORDS = [
    "sad",
    "lost",
    "broken",
    "hospital",
    "bad ending",
    "never",
    "could not",
    "failed",
    "alone",
    "cried",
    "cry",
    "gone",
    "hurt",
    "scared",
    "afraid",
    "sick",
    "sorry",
    "fell",
    "broke",
    "would not",
    "did not",
]


def resolve_project_path(script_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = script_dir / path
    return path


def normalize_space(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def split_sentences(text: str) -> list[str]:
    # TinyStories is English text, so a simple sentence splitter is enough for this homework dataset.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def clean_story_text(raw_text: str, take_part: str, max_sentences: int, max_chars: int) -> str:
    """Clean a TinyStories story and keep a short span for ending-polarity labeling."""

    text = normalize_space(raw_text)
    sentences = split_sentences(text)
    if not sentences:
        return ""

    if take_part == "beginning":
        selected = sentences[:max_sentences]
    elif take_part == "full_short":
        selected = sentences[:max_sentences]
    elif take_part == "ending":
        selected = sentences[-max_sentences:]
    else:
        raise ValueError(f"Unsupported take_part: {take_part}")

    cleaned = " ".join(selected).strip()
    if len(cleaned) > max_chars:
        # Keep whole sentences if possible; otherwise fall back to a clean character cut.
        shortened: list[str] = []
        current_len = 0
        for sentence in selected:
            extra = len(sentence) + (1 if shortened else 0)
            if current_len + extra > max_chars:
                break
            shortened.append(sentence)
            current_len += extra
        cleaned = " ".join(shortened).strip() if shortened else cleaned[:max_chars].rsplit(" ", 1)[0].strip()
    return cleaned


def looks_broken(text: str) -> bool:
    if "\ufffd" in text:
        return True
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text):
        return True
    letters = sum(ch.isalpha() for ch in text)
    return letters < 20


def fallback_token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", text))


def load_optional_tokenizer(script_dir: Path, vocab_path_value: str):
    if not vocab_path_value:
        return None
    from tiny_tokenizer import load_tokenizer

    vocab_path = resolve_project_path(script_dir, vocab_path_value)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Cannot find vocab_path: {vocab_path}")
    return load_tokenizer(vocab_path)


def count_tokens(text: str, tokenizer) -> int:
    if tokenizer is None:
        return fallback_token_count(text)
    return len(tokenizer.encode(text))


def parse_keywords(args: argparse.Namespace) -> list[str]:
    keywords: list[str] = []
    if args.keyword_preset == "negative_endings":
        keywords.extend(NEGATIVE_ENDING_KEYWORDS)
    if args.keywords:
        keywords.extend(part.strip().lower() for part in args.keywords.split(",") if part.strip())
    # Keep order while removing duplicates.
    seen: set[str] = set()
    unique_keywords = []
    for keyword in keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)
    return unique_keywords


def matched_keywords(text: str, keywords: list[str]) -> list[str]:
    text_lower = text.lower()
    hits = []
    for keyword in keywords:
        if " " in keyword:
            if keyword in text_lower:
                hits.append(keyword)
        elif re.search(rf"\b{re.escape(keyword)}\b", text_lower):
            hits.append(keyword)
    return hits


def load_tinystories_dataset(args: argparse.Namespace, script_dir: Path):
    if args.load_mode == "hub":
        from datasets import load_dataset

        split_expr = f"{args.split}[{args.start_index}:]"
        return load_dataset(args.dataset_name, split=split_expr), args.start_index

    if args.load_mode == "disk":
        from datasets import load_from_disk

        disk_path = resolve_project_path(script_dir, args.hf_disk_path)
        if not disk_path.exists():
            raise FileNotFoundError(f"Cannot find local Hugging Face dataset path: {disk_path}")
        return load_from_disk(str(disk_path)), args.start_index

    raise ValueError(f"Unsupported load_mode: {args.load_mode}")


def reservoir_add(items: list[dict[str, Any]], item: dict[str, Any], seen: int, limit: int, rng: random.Random) -> None:
    if len(items) < limit:
        items.append(item)
        return
    j = rng.randint(0, seen - 1)
    if j < limit:
        items[j] = item


def generate_labeling_guide(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    guide = """# Fine-tuning Labeling Guide

## Task

`story_ending_polarity_classification`

Label the ending tendency of each short TinyStories text.

## Labels

`positive`: The ending is positive. A problem is solved, a character receives help, a conflict is softened, or the character ends safe, happy, or with a clear gain.

`negative`: The ending is negative. A problem remains unsolved, a character fails, feels lonely, afraid, or sad, a conflict is not repaired, or the bad result continues.

`discard`: The text is too short, too unclear, lacks an ending, has no obvious polarity, or is not suitable for the experiment.

## Principles

- Only use the information shown in the `text` field.
- Focus on the ending tendency, not just one emotional word.
- If there is a difficulty and it is solved at the end, label `positive`.
- If there is a difficulty and it is not solved at the end, label `negative`.
- If the text is neutral or impossible to judge, label `discard`.

## Examples

Positive:

> Tom lost his red ball. Mia found it under the bench and gave it back. They played together until dinner.

Negative:

> Tom lost his red ball. He looked under the bench but it was gone. He walked home with empty hands.

Discard:

> Tom had a ball. It was red.
"""
    path.write_text(guide, encoding="utf-8")


def extract_candidates(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    output_path = resolve_project_path(script_dir, args.output_path)
    guide_path = resolve_project_path(script_dir, args.labeling_guide_path)
    tokenizer = load_optional_tokenizer(script_dir, args.vocab_path)
    keywords = parse_keywords(args)
    rng = random.Random(args.seed)

    dataset, source_start = load_tinystories_dataset(args, script_dir)
    candidates: list[dict[str, Any]] = []
    seen_eligible = 0
    seen_texts: set[str] = set()
    scanned = 0

    for local_index, item in enumerate(dataset):
        if args.scan_limit > 0 and scanned >= args.scan_limit:
            break
        scanned += 1

        raw_text = str(item.get(args.text_field, "")).strip()
        text = clean_story_text(raw_text, args.take_part, args.max_sentences, args.max_chars)
        if len(text) < args.min_chars or looks_broken(text):
            continue
        if text in seen_texts:
            continue

        hits = matched_keywords(text, keywords)
        if args.require_keywords and keywords and not hits:
            continue

        n_tokens = count_tokens(text, tokenizer)
        notes_parts = []
        if hits:
            notes_parts.append("keywords=" + "|".join(hits))
        if n_tokens > args.max_length:
            if args.drop_too_long:
                continue
            notes_parts.append("too_long")

        seen_texts.add(text)
        seen_eligible += 1
        source_index = source_start + local_index
        row = {
            "id": "",  # Filled after sampling so ids are compact and stable.
            "text": text,
            "label": "",
            "n_tokens": n_tokens,
            "n_chars": len(text),
            "source_split": args.split,
            "source_index": source_index,
            "notes": ";".join(notes_parts),
        }
        reservoir_add(candidates, row, seen_eligible, args.num_samples, rng)

    candidates.sort(key=lambda row: row["source_index"])
    for i, row in enumerate(candidates, start=1):
        row["id"] = f"ts_{i:06d}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["id", "text", "label", "n_tokens", "n_chars", "source_split", "source_index", "notes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    generate_labeling_guide(guide_path)
    print(f"Scanned samples: {scanned}")
    print(f"Eligible samples: {seen_eligible}")
    print(f"Saved candidates: {len(candidates)}")
    if keywords:
        print(f"Keyword preset: {args.keyword_preset}")
        print(f"Require keywords: {args.require_keywords}")
        print(f"Keywords: {', '.join(keywords)}")
    print(f"Candidates CSV: {output_path}")
    print(f"Labeling guide: {guide_path}")
    if len(candidates) < args.num_samples:
        print("Warning: fewer candidates were found than requested. Try increasing --scan_limit or relaxing filters.")


def read_annotation_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    required = {"text", "label"}
    if not rows or not required.issubset(rows[0].keys()):
        raise ValueError("Annotation CSV must contain at least text,label columns.")
    return rows


def make_csv(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    input_path = resolve_project_path(script_dir, args.input_path)
    output_path = resolve_project_path(script_dir, args.output_path)
    rows = read_annotation_rows(input_path)
    rng = random.Random(args.seed)

    counts = {"positive": 0, "negative": 0, "discard": 0, "empty": 0, "invalid": 0}
    usable_by_label: dict[str, list[dict[str, str]]] = {"positive": [], "negative": []}
    seen_texts: set[str] = set()
    duplicate_count = 0

    for row in rows:
        label = row.get("label", "").strip().lower()
        text = row.get("text", "").strip()
        if label not in VALID_LABELS:
            counts["invalid"] += 1
            print(f"Warning: invalid label skipped: {label!r}")
            continue
        if label == "":
            counts["empty"] += 1
            continue
        counts[label] += 1
        if label not in FINAL_LABELS:
            continue
        if text in seen_texts:
            duplicate_count += 1
            continue
        seen_texts.add(text)
        usable_by_label[label].append({"text": text, "label": label})

    if args.balance:
        keep = min(len(usable_by_label["positive"]), len(usable_by_label["negative"]))
        for label in FINAL_LABELS:
            rng.shuffle(usable_by_label[label])
            usable_by_label[label] = usable_by_label[label][:keep]

    final_rows = usable_by_label["positive"] + usable_by_label["negative"]
    rng.shuffle(final_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(final_rows)

    positive_count = sum(1 for row in final_rows if row["label"] == "positive")
    negative_count = sum(1 for row in final_rows if row["label"] == "negative")
    print(f"Total candidates: {len(rows)}")
    print(f"Annotated positive: {counts['positive']}")
    print(f"Annotated negative: {counts['negative']}")
    print(f"Discard: {counts['discard']}")
    print(f"Empty label: {counts['empty']}")
    print(f"Invalid label: {counts['invalid']}")
    print(f"Duplicate usable text removed: {duplicate_count}")
    print(f"Final positive: {positive_count}")
    print(f"Final negative: {negative_count}")
    print(f"Final usable samples: {len(final_rows)}")
    print(f"Output CSV: {output_path}")
    if positive_count != negative_count:
        print("Warning: positive and negative counts are not balanced. Use --balance if needed.")
    if len(final_rows) < 200:
        print("Warning: final usable samples are fewer than 200.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a manually labeled fine-tuning CSV from TinyStories.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Extract TinyStories candidates for manual labeling.")
    extract.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
    extract.add_argument("--split", type=str, default="train")
    extract.add_argument("--start_index", type=int, default=5000)
    extract.add_argument("--num_samples", type=int, default=300)
    extract.add_argument("--max_chars", type=int, default=500)
    extract.add_argument("--min_chars", type=int, default=80)
    extract.add_argument("--max_sentences", type=int, default=3)
    extract.add_argument("--seed", type=int, default=123)
    extract.add_argument("--output_path", type=str, default="data/finetune_annotation_candidates.csv")
    extract.add_argument("--labeling_guide_path", type=str, default="data/finetune_labeling_guide.md")
    extract.add_argument("--vocab_path", type=str, default="")
    extract.add_argument("--max_length", type=int, default=64)
    extract.add_argument("--take_part", type=str, default="ending", choices=["beginning", "ending", "full_short"])
    extract.add_argument("--load_mode", type=str, default="hub", choices=["hub", "disk"])
    extract.add_argument("--hf_disk_path", type=str, default="data/TinyStories_train")
    extract.add_argument("--text_field", type=str, default="text")
    extract.add_argument("--scan_limit", type=int, default=5000)
    extract.add_argument("--drop_too_long", action="store_true")
    extract.add_argument("--keyword_preset", type=str, default="none", choices=["none", "negative_endings"])
    extract.add_argument("--keywords", type=str, default="", help="Comma-separated extra keywords or phrases.")
    extract.add_argument("--require_keywords", action="store_true", help="Keep only candidates containing at least one keyword.")
    extract.add_argument("--cpu_safe", action="store_true", help="Compatibility flag; extraction does not use GPU or multiprocessing.")
    extract.set_defaults(func=extract_candidates)

    make = subparsers.add_parser("make_csv", help="Validate manual labels and create data/finetune.csv.")
    make.add_argument("--input_path", type=str, default="data/finetune_annotation_candidates.csv")
    make.add_argument("--output_path", type=str, default="data/finetune.csv")
    make.add_argument("--seed", type=int, default=123)
    make.add_argument("--balance", action="store_true")
    make.set_defaults(func=make_csv)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
