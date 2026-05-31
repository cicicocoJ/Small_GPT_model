"""Tokenizer utilities for the tiny GPT project.

This file supports two tokenizers:
1. GPT2Tokenizer: GPT-2 BPE tokenizer from tiktoken.
2. SimpleRegexTokenizer: a small corpus tokenizer based on regex tokens.

For this homework, regex is usually better for TinyStories experiments because
it keeps the vocabulary small and makes the output head much smaller.
"""

import json
import re
from pathlib import Path
from typing import Any


class GPT2Tokenizer:
    """Light wrapper around tiktoken's GPT-2 tokenizer."""

    tokenizer_type = "gpt2"
    encoding_name = "gpt2"
    eos_token = "<|endoftext|>"

    def __init__(self, encoding_name: str = "gpt2"):
        self.encoding_name = encoding_name
        import tiktoken

        self.tokenizer = tiktoken.get_encoding(encoding_name)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.n_vocab

    @property
    def eos_id(self) -> int:
        return self.tokenizer.eot_token

    @property
    def pad_id(self) -> int:
        return self.eos_id

    @classmethod
    def build_from_text(cls, text: str) -> "GPT2Tokenizer":
        _ = text
        return cls()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, allowed_special={self.eos_token})

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        ids = [int(token_id) for token_id in token_ids]
        if skip_special_tokens:
            ids = [token_id for token_id in ids if token_id != self.eos_id]
        return self.tokenizer.decode(ids)

    def save(self, config_path: str | Path) -> None:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "tokenizer_type": self.tokenizer_type,
            "encoding_name": self.encoding_name,
            "vocab_size": self.vocab_size,
            "eos_token": self.eos_token,
            "eos_id": self.eos_id,
            "pad_id": self.pad_id,
        }
        config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, config_path: str | Path) -> "GPT2Tokenizer":
        data = json.loads(Path(config_path).read_text(encoding="utf-8"))
        return cls(encoding_name=data.get("encoding_name", "gpt2"))


class SimpleRegexTokenizer:
    """Small regex tokenizer for tiny English corpora.

    The tokenizer treats <|endoftext|> as one complete special token, then
    splits normal text into words, numbers, punctuation, and newlines.
    """

    tokenizer_type = "regex"
    eos_token = "<|endoftext|>"
    pad_token = "<PAD>"
    unk_token = "<UNK>"
    newline_token = "\n"
    token_pattern = re.compile(r"<\|endoftext\|>|\n|--|[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", re.UNICODE)

    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {idx: token for token, idx in token_to_id.items()}
        if self.eos_token not in token_to_id or self.unk_token not in token_to_id:
            raise ValueError("regex tokenizer vocabulary must contain <|endoftext|> and <UNK>")

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def pad_id(self) -> int:
        return self.eos_id

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        return cls.token_pattern.findall(text)

    @classmethod
    def build_from_text(cls, text: str) -> "SimpleRegexTokenizer":
        tokens = cls.tokenize(text)
        unique_tokens = sorted(set(tokens))
        token_to_id = {
            cls.eos_token: 0,
            cls.unk_token: 1,
            cls.pad_token: 2,
        }
        for token in unique_tokens:
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)
        return cls(token_to_id)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in self.tokenize(text)]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id), self.unk_token)
            if token == self.pad_token:
                continue
            if skip_special_tokens and token == self.eos_token:
                continue
            tokens.append(token)

        text = ""
        no_space_before = {".", ",", ":", ";", "?", "!", ")", "]", "}", "'", '"'}
        no_space_after = {"(", "[", "{", '"'}

        for token in tokens:
            if token == self.newline_token:
                text = text.rstrip() + "\n"
            elif token == self.eos_token:
                text = text.rstrip() + self.eos_token
            elif not text or text.endswith(("\n", " ")) or token in no_space_before:
                text += token
            elif text[-1] in no_space_after:
                text += token
            else:
                text += " " + token
        return text

    def save(self, config_path: str | Path) -> None:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "tokenizer_type": self.tokenizer_type,
            "token_to_id": self.token_to_id,
            "eos_token": self.eos_token,
            "eos_id": self.eos_id,
            "pad_token": self.pad_token,
            "pad_id": self.pad_id,
            "unk_token": self.unk_token,
            "vocab_size": self.vocab_size,
        }
        config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, config_path: str | Path) -> "SimpleRegexTokenizer":
        data = json.loads(Path(config_path).read_text(encoding="utf-8"))
        token_to_id = {token: int(idx) for token, idx in data["token_to_id"].items()}
        return cls(token_to_id)


def build_tokenizer(tokenizer_name: str, text: str):
    """Build tokenizer by name."""

    if tokenizer_name == "gpt2":
        return GPT2Tokenizer.build_from_text(text)
    if tokenizer_name == "regex":
        return SimpleRegexTokenizer.build_from_text(text)
    raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")


def load_tokenizer(config_path: str | Path):
    """Load tokenizer from saved config file."""

    data: dict[str, Any] = json.loads(Path(config_path).read_text(encoding="utf-8"))
    tokenizer_type = data.get("tokenizer_type", "gpt2")
    if tokenizer_type in {"gpt2", "tiktoken_gpt2"}:
        return GPT2Tokenizer.load(config_path)
    if tokenizer_type == "regex":
        return SimpleRegexTokenizer.load(config_path)
    raise ValueError(f"Unsupported tokenizer type in config: {tokenizer_type}")


# Backward-compatible alias for older imports.
CharTokenizer = SimpleRegexTokenizer


def main() -> None:
    sample_text = "Once upon a time<|endoftext|>A tiny GPT learned words.\nIt was small, but it worked!"
    tokenizer = SimpleRegexTokenizer.build_from_text(sample_text)
    ids = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(ids)
    decoded_without_special = tokenizer.decode(ids, skip_special_tokens=True)

    print("Tokenizer type: regex")
    print("Vocab size:", tokenizer.vocab_size)
    print("EOS token id:", tokenizer.eos_id)
    print("PAD token id:", tokenizer.pad_id)
    print("Tokens:", tokenizer.tokenize(sample_text))
    print("Encoded ids:", ids)
    print("Decoded text:", decoded)
    print("Decoded without special:", decoded_without_special)

    assert SimpleRegexTokenizer.eos_token in tokenizer.token_to_id
    assert tokenizer.tokenize("a<|endoftext|>b") == ["a", SimpleRegexTokenizer.eos_token, "b"]
    assert "< | endoftext | >" not in decoded
    assert SimpleRegexTokenizer.eos_token not in decoded_without_special
    print("Tokenizer self-test passed.")


if __name__ == "__main__":
    main()
