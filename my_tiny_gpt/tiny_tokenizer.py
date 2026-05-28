"""字符级 tokenizer。

本作业重点是跑通小型 GPT 的完整流程，因此这里使用最简单稳定的字符级分词：
每个字符对应一个 token id。它不追求工业级分词效果，但非常适合本科课程展示。
"""

import json
from pathlib import Path


class CharTokenizer:
    """简单字符级 tokenizer。

    功能：
    1. 根据训练语料构建字符词表；
    2. 将文本编码为 token ids；
    3. 将 token ids 解码回文本；
    4. 保存和加载词表。
    """

    pad_token = "<PAD>"
    unk_token = "<UNK>"

    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {idx: token for token, idx in token_to_id.items()}

        if self.pad_token not in self.token_to_id:
            raise ValueError("词表中必须包含 <PAD>")
        if self.unk_token not in self.token_to_id:
            raise ValueError("词表中必须包含 <UNK>")

    @property
    def vocab_size(self) -> int:
        """返回词表大小，用于初始化 MiniGPT 的 vocab_size。"""

        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        """返回 padding token 的 id。"""

        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        """返回未知字符 token 的 id。"""

        return self.token_to_id[self.unk_token]

    @classmethod
    def build_from_text(cls, text: str) -> "CharTokenizer":
        """从一段文本中构建字符词表。

        为了结果可复现，普通字符按排序后的顺序加入词表。
        """

        chars = sorted(set(text))
        token_to_id = {
            cls.pad_token: 0,
            cls.unk_token: 1,
        }

        for char in chars:
            if char not in token_to_id:
                token_to_id[char] = len(token_to_id)

        return cls(token_to_id)

    @classmethod
    def build_from_file(cls, text_path: str | Path) -> "CharTokenizer":
        """从文本文件读取语料并构建词表。"""

        text_path = Path(text_path)
        text = text_path.read_text(encoding="utf-8")
        return cls.build_from_text(text)

    def encode(self, text: str) -> list[int]:
        """将字符串编码为 token id 列表。"""

        return [self.token_to_id.get(char, self.unk_id) for char in text]

    def decode(self, token_ids: list[int]) -> str:
        """将 token id 列表解码回字符串。

        解码时跳过 <PAD>，遇到未知 id 时用 <UNK> 占位。
        """

        chars = []
        for token_id in token_ids:
            token = self.id_to_token.get(int(token_id), self.unk_token)
            if token == self.pad_token:
                continue
            chars.append(token)
        return "".join(chars)

    def save(self, vocab_path: str | Path) -> None:
        """保存词表到 JSON 文件，后续训练和推理共用同一个词表。"""

        vocab_path = Path(vocab_path)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "token_to_id": self.token_to_id,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
        }
        vocab_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, vocab_path: str | Path) -> "CharTokenizer":
        """从 JSON 文件加载词表。"""

        vocab_path = Path(vocab_path)
        data = json.loads(vocab_path.read_text(encoding="utf-8"))
        token_to_id = {token: int(idx) for token, idx in data["token_to_id"].items()}
        return cls(token_to_id)


def main() -> None:
    """命令行自测：构建词表、保存词表、测试 encode/decode。"""

    project_dir = Path(__file__).resolve().parent
    text_path = project_dir / "data" / "pretrain.txt"
    vocab_path = project_dir / "outputs" / "vocab.json"

    tokenizer = CharTokenizer.build_from_file(text_path)
    tokenizer.save(vocab_path)

    sample_text = "Learning begins with curiosity."
    token_ids = tokenizer.encode(sample_text)
    decoded_text = tokenizer.decode(token_ids)

    print("语料路径:", text_path)
    print("词表路径:", vocab_path)
    print("词表大小:", tokenizer.vocab_size)
    print("样例文本:", sample_text)
    print("编码结果:", token_ids)
    print("解码结果:", decoded_text)
    assert decoded_text == sample_text
    print("测试通过：字符级 tokenizer 可以正常编码和解码。")


if __name__ == "__main__":
    main()
