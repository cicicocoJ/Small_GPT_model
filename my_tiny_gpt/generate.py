"""加载预训练 MiniGPT 并生成文本。

运行示例（Windows VS Code 终端）：
    python .\\my_tiny_gpt\\generate.py

自定义生成参数：
    python .\\my_tiny_gpt\\generate.py --prompt "Once upon a time" --max_new_tokens 100 --temperature 0.7 --top_k 20
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from mini_gpt import MiniGPT, MiniGPTConfig
from tiny_tokenizer import CharTokenizer


DEFAULT_PROMPTS = [
    "Once upon a time",
    "The student learned",
    "A language model",
]


def load_checkpoint(model_path: Path, device: torch.device) -> dict:
    """加载预训练 checkpoint。

    checkpoint 由 train_pretrain.py 保存，应该包含：
    - model_state_dict: 模型权重
    - config: 模型配置
    - vocab_path: 词表路径
    """

    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # 兼容旧版本 PyTorch：旧版本 torch.load 没有 weights_only 参数。
        return torch.load(model_path, map_location=device)


def build_model_from_checkpoint(checkpoint: dict, device: torch.device) -> MiniGPT:
    """根据 checkpoint 中保存的配置重建模型并加载权重。"""

    if "config" not in checkpoint:
        raise KeyError("checkpoint 中缺少 config，请重新运行 train_pretrain.py 保存模型配置。")
    if "model_state_dict" not in checkpoint:
        raise KeyError("checkpoint 中缺少 model_state_dict，请检查预训练权重文件。")

    cfg = MiniGPTConfig(**checkpoint["config"])
    model = MiniGPT(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # 生成前切换到 eval 模式，关闭 dropout 等训练行为。
    model.eval()
    return model


def load_tokenizer(checkpoint: dict, default_vocab_path: Path) -> CharTokenizer:
    """加载训练时保存的 tokenizer 词表。"""

    vocab_path = default_vocab_path
    checkpoint_vocab_path = checkpoint.get("vocab_path")

    # 如果 checkpoint 中记录的路径存在，则优先使用；否则使用 outputs/vocab.json。
    if checkpoint_vocab_path:
        candidate = Path(checkpoint_vocab_path)
        if candidate.exists():
            vocab_path = candidate

    if not vocab_path.exists():
        raise FileNotFoundError(
            f"找不到词表文件: {vocab_path}。请先运行 tiny_tokenizer.py 或 train_pretrain.py。"
        )

    return CharTokenizer.load(vocab_path)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    """根据 temperature 和 top_k 从最后一步 logits 中采样下一个 token。"""

    # temperature <= 0 时使用贪心解码，直接选择概率最高的 token。
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    # top_k > 0 时，只保留概率最高的 k 个 token，降低随机乱跳的概率。
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        top_values, _ = torch.topk(logits, top_k)
        min_top_value = top_values[:, -1].unsqueeze(-1)
        logits = torch.where(
            logits < min_top_value,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_text(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> str:
    """自回归生成文本。

    每一步都把当前上下文输入模型，取最后一个位置的 logits 采样下一个 token，
    然后把新 token 拼接回输入序列，循环生成直到达到 max_new_tokens。
    """

    if not prompt:
        raise ValueError("prompt 不能为空")

    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 超过模型最大上下文长度时，只保留最近的 context_length 个 token。
            input_cond = input_ids[:, -model.cfg.context_length :]
            logits = model(input_cond)
            next_token_logits = logits[:, -1, :]
            next_token_id = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
            )
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

    return tokenizer.decode(input_ids.squeeze(0).tolist())


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Generate text with pretrained MiniGPT.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="预训练模型权重路径，默认使用 my_tiny_gpt/outputs/pretrain_model.pt",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=None,
        help="词表路径，默认使用 my_tiny_gpt/outputs/vocab.json",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="输入 prompt。可重复传入多次；不传则使用 3 个默认测试 prompt。",
    )
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return parser.parse_args()


def main() -> None:
    """加载模型和词表，并对 prompt 进行生成测试。"""

    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "outputs"
    model_path = Path(args.model_path) if args.model_path else output_dir / "pretrain_model.pt"
    vocab_path = Path(args.vocab_path) if args.vocab_path else output_dir / "vocab.json"
    result_path = output_dir / "generated_samples.txt"

    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型权重: {model_path}。请先运行 train_pretrain.py。")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    checkpoint = load_checkpoint(model_path, device)
    tokenizer = load_tokenizer(checkpoint, vocab_path)
    model = build_model_from_checkpoint(checkpoint, device)

    prompts = args.prompt if args.prompt else DEFAULT_PROMPTS
    result_lines = [
        f"model_path: {model_path}",
        f"vocab_size: {tokenizer.vocab_size}",
        f"device: {device}",
        f"max_new_tokens: {args.max_new_tokens}",
        f"temperature: {args.temperature}",
        f"top_k: {args.top_k}",
        "",
    ]

    for index, prompt in enumerate(prompts, start=1):
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )

        print(f"\n===== Sample {index} =====")
        print(f"Prompt: {prompt}")
        print("Generated:")
        print(generated)

        result_lines.extend(
            [
                f"===== Sample {index} =====",
                f"Prompt: {prompt}",
                "Generated:",
                generated,
                "",
            ]
        )

    result_path.write_text("\n".join(result_lines), encoding="utf-8")
    print(f"\n生成结果已保存到: {result_path}")


if __name__ == "__main__":
    main()
