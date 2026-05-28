"""小型 GPT 模型。

本文件参考 ch04 的 GPT 模型结构，但使用更小的参数规模，
适合课程作业在普通电脑上完成训练和推理演示。
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MiniGPTConfig:
    """小型 GPT 的默认配置。"""

    vocab_size: int = 1000
    context_length: int = 128
    emb_dim: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    qkv_bias: bool = False


class CausalSelfAttention(nn.Module):
    """因果多头自注意力层。

    因果 mask 会遮住当前位置之后的 token，保证模型只能根据历史文本预测下一个 token。
    """

    def __init__(self, cfg: MiniGPTConfig):
        super().__init__()
        assert cfg.emb_dim % cfg.n_heads == 0, "emb_dim 必须能被 n_heads 整除"

        self.n_heads = cfg.n_heads
        self.head_dim = cfg.emb_dim // cfg.n_heads
        self.emb_dim = cfg.emb_dim

        # 将输入分别映射为 query、key、value。
        self.W_query = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)
        self.W_key = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)
        self.W_value = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)

        # 多头注意力拼接后再做一次线性投影。
        self.out_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # 上三角矩阵表示“未来位置”，注册为 buffer 后会随模型一起移动到 CPU/GPU。
        mask = torch.triu(torch.ones(cfg.context_length, cfg.context_length), diagonal=1)
        self.register_buffer("mask", mask.bool())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # [B, T, C] -> [B, T, n_heads, head_dim] -> [B, n_heads, T, head_dim]
        queries = self.W_query(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        keys = self.W_key(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        values = self.W_value(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # 缩放点积注意力。
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores / (self.head_dim**0.5)

        # 应用因果 mask，避免看到未来 token。
        current_mask = self.mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(current_mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 使用注意力权重汇总 value，并把多个头重新合并。
        context = attn_weights @ values
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.emb_dim)

        out = self.out_proj(context)
        return self.resid_dropout(out)


class FeedForward(nn.Module):
    """Transformer block 中的前馈网络。"""

    def __init__(self, cfg: MiniGPTConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    """一个 GPT Transformer block。"""

    def __init__(self, cfg: MiniGPTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.emb_dim)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.emb_dim)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 注意力子层：LayerNorm -> causal self-attention -> residual connection。
        x = x + self.attn(self.norm1(x))

        # 前馈子层：LayerNorm -> feed forward -> residual connection。
        x = x + self.ff(self.norm2(x))
        return x


class MiniGPT(nn.Module):
    """小型 GPT 语言模型。

    输入形状为 [batch_size, seq_len] 的 token ids，
    输出形状为 [batch_size, seq_len, vocab_size] 的 logits。
    """

    def __init__(self, cfg: MiniGPTConfig):
        super().__init__()
        self.cfg = cfg

        # token embedding 表示 token 本身，position embedding 表示 token 的位置。
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.dropout)

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = nn.LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = token_ids.shape
        if seq_len > self.cfg.context_length:
            raise ValueError(f"输入长度 {seq_len} 超过 context_length={self.cfg.context_length}")

        positions = torch.arange(seq_len, device=token_ids.device)
        tok_embeds = self.tok_emb(token_ids)
        pos_embeds = self.pos_emb(positions)

        x = self.drop_emb(tok_embeds + pos_embeds)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量，方便写入实验报告。"""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def simple_test() -> None:
    """简单测试：随机输入 token ids，确认模型可以输出 logits。"""

    torch.manual_seed(123)
    cfg = MiniGPTConfig()
    model = MiniGPT(cfg)

    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(input_ids)

    print("输入 token ids 形状:", tuple(input_ids.shape))
    print("输出 logits 形状:", tuple(logits.shape))
    print("可训练参数量:", count_parameters(model))

    expected_shape = (2, 16, cfg.vocab_size)
    assert logits.shape == expected_shape, f"期望 {expected_shape}, 实际 {tuple(logits.shape)}"
    print("测试通过：输入 token ids 后可以正常输出 logits。")


if __name__ == "__main__":
    simple_test()
