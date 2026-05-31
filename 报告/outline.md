# 基于 PyTorch 的小型 GPT 模型构建与训练

## 摘要

本文围绕“从零构建大模型”课程任务，基于 PyTorch 设计并实现了一个小参数量 GPT 类语言模型。项目主要包括 GPT 模型结构设计、文本分词与数据处理、预训练程序实现、文本生成测试以及特定任务微调等内容。实验采用适合低算力平台的小型模型配置，通过正则分词或 GPT-2 BPE 分词将文本转换为 token 序列，并使用 next-token prediction 作为预训练目标。实验结果表明，小型 GPT 模型虽然无法达到真实大语言模型的生成质量，但能够较完整地体现文本编码、词嵌入、位置编码、因果自注意力、预训练和微调等核心机制。

**关键词：** GPT；Transformer；PyTorch；预训练；微调；Tokenizer

---

# 目录

* [1 绪论](#1-绪论)

  * [1.1 项目背景与意义](#11-项目背景与意义)
  * [1.2 项目任务与技术路线](#12-项目任务与技术路线)

* [2 大语言模型相关理论基础](#2-大语言模型相关理论基础)

  * [2.1 GPT 类语言模型基本原理](#21-gpt-类语言模型基本原理)
  * [2.2 文本分词与输入表示](#22-文本分词与输入表示)
  * [2.3 Transformer 核心结构](#23-transformer-核心结构)
  * [2.4 预训练与微调机制](#24-预训练与微调机制)

* [3 小型 GPT 模型设计与实现](#3-小型-gpt-模型设计与实现)

  * [3.1 模型总体结构](#31-模型总体结构)
  * [3.2 核心模块设计](#32-核心模块设计)
  * [3.3 模型参数配置](#33-模型参数配置)

* [4 数据处理与预训练实验](#4-数据处理与预训练实验)

  * [4.1 预训练数据来源与处理](#41-预训练数据来源与处理)
  * [4.2 Tokenizer 设计与对比](#42-tokenizer-设计与对比)
  * [4.3 预训练任务与训练策略](#43-预训练任务与训练策略)
  * [4.4 预训练结果与生成效果分析](#44-预训练结果与生成效果分析)

* [5 微调实验与任务适应](#5-微调实验与任务适应)

  * [5.1 微调任务与数据构建](#51-微调任务与数据构建)
  * [5.2 微调方法设计](#52-微调方法设计)
  * [5.3 微调结果与分析](#53-微调结果与分析)

* [6 总结与体会](#6-总结与体会)

  * [6.1 项目完成情况](#61-项目完成情况)
  * [6.2 实验问题与改进方向](#62-实验问题与改进方向)
  * [6.3 学习收获与体会](#63-学习收获与体会)

* [参考文献与开源资源说明](#参考文献与开源资源说明)

* [附录](#附录)

---

# 1 绪论

## 1.1 项目背景与意义

简要介绍大语言模型和 GPT 类模型的发展背景，说明 GPT 模型在文本生成和自然语言处理任务中的代表性。

说明本项目不是训练真正的大规模商用模型，而是在低算力条件下实现一个小型 GPT 模型，通过完整流程理解大语言模型的基本原理。

## 1.2 项目任务与技术路线

概括课程作业要求，包括知识点梳理、模型搭建、预训练、微调和结果展示。

配合技术路线图说明整体流程：文本数据准备、Tokenizer 分词、训练样本构造、小型 GPT 模型搭建、预训练、文本生成测试、监督数据构建、模型微调、结果分析。

---

# 2 大语言模型相关理论基础

## 2.1 GPT 类语言模型基本原理

介绍大语言模型的基本概念，说明 GPT 属于 Decoder-only Transformer 结构，是一种自回归语言模型。

### 2.1.1 自回归语言建模

说明 GPT 通过已有上下文预测下一个 token，训练目标是 next-token prediction。

### 2.1.2 GPT 的基本计算流程

说明输入 token 序列经过 embedding、Transformer Block 和输出层，最终得到下一个 token 的概率分布。

## 2.2 文本分词与输入表示

说明自然语言文本需要先转换为 token id，再输入神经网络。

### 2.2.1 Tokenizer 的作用

说明 tokenizer 用于完成文本切分、词表构建、编码和解码。

### 2.2.2 正则分词方法

说明本实验主要采用基于正则表达式的简单 tokenizer，将文本切分为英文单词、数字、标点和换行符等 token。该方法比字符级分词更容易保留完整单词，同时词表规模远小于 GPT-2 BPE 词表，适合小语料实验。

### 2.2.3 GPT-2 BPE 分词方法

说明代码中也提供 GPT-2 BPE tokenizer 作为可选方案。该方法更接近真实 GPT 模型使用的子词分词方式，但词表规模较大，对小模型和低算力训练不一定友好。

### 2.2.4 词嵌入与位置嵌入

说明 token embedding 用于表示 token 本身，position embedding 用于表示 token 在序列中的位置，两者相加后作为 Transformer 的输入。

## 2.3 Transformer 核心结构

说明 Transformer 是 GPT 类模型的核心结构。

### 2.3.1 因果多头自注意力

说明多头自注意力用于建模不同 token 之间的上下文关系，因果 mask 用于防止模型看到未来 token。

### 2.3.2 前馈网络、残差连接与层归一化

说明每个 Transformer Block 中包含注意力子层和前馈网络子层，并通过残差连接和 LayerNorm 稳定训练过程。

## 2.4 预训练与微调机制

说明预训练和微调的区别。

### 2.4.1 预训练

预训练使用无标签文本数据，通过预测下一个 token 让模型学习基本语言模式。

### 2.4.2 微调

微调在预训练模型基础上，使用带标签或任务格式数据继续训练，使模型适应特定任务。

---

# 3 小型 GPT 模型设计与实现

## 3.1 模型总体结构

本实验实现的小型 GPT 模型采用 Decoder-only Transformer 架构，主要由 token embedding、position embedding、多层 Transformer Block、最终 LayerNorm 和线性输出层组成。

模型整体流程为：

```text
输入 token ids
→ Token Embedding
→ Position Embedding
→ Transformer Block × N
→ Final LayerNorm
→ Linear Head
→ logits
```

## 3.2 核心模块设计

介绍 `mini_gpt.py` 中的主要模块。

### 3.2.1 MiniGPTConfig

说明模型配置类中包含 vocab_size、context_length、emb_dim、n_heads、n_layers、dropout 等参数。

### 3.2.2 CausalSelfAttention

说明该模块实现因果多头自注意力，通过上三角 mask 屏蔽未来位置。

### 3.2.3 TransformerBlock

说明每个 Transformer Block 由 LayerNorm、因果自注意力、前馈网络和残差连接组成。

### 3.2.4 MiniGPT 输出层

说明模型输出形状为 `[batch_size, seq_len, vocab_size]`，用于计算每个位置上的 next-token prediction loss。

## 3.3 模型参数配置

说明本实验为了适应普通电脑训练，采用小参数配置。

| 参数名称           | 含义             | 设置值               |
| -------------- | -------------- | ----------------- |
| vocab_size     | 词表大小           | 根据 tokenizer 自动确定 |
| context_length | 上下文长度          | 待填写               |
| emb_dim        | 词嵌入维度          | 128               |
| n_heads        | 注意力头数          | 4                 |
| n_layers       | Transformer 层数 | 4                 |
| dropout        | Dropout 比例     | 0.1               |
| 参数量            | 可训练参数量         | 待填写               |

---

# 4 数据处理与预训练实验

## 4.1 预训练数据来源与处理

说明本实验支持两类数据来源：本地小语料和 TinyStories 数据集。为了先完成低算力流程验证，可以使用本地英文短文；为了获得更明显的训练效果，也可以使用 TinyStories 的一小部分数据。

### 4.1.1 本地文本数据

说明本地数据文件路径、文本内容类型和规模。

### 4.1.2 TinyStories 数据集

说明训练脚本支持从 Hugging Face 加载 TinyStories，并可指定训练集和验证集切片，适合小模型学习简单英文故事模式。

### 4.1.3 文本清洗

说明程序中可以进行换行统一、连续空行压缩等简单清洗操作，使语料格式更稳定。

## 4.2 Tokenizer 设计与对比

本实验代码提供正则分词和 GPT-2 BPE 分词两种选择，默认使用正则分词。

### 4.2.1 正则分词 tokenizer

说明正则 tokenizer 将文本切分为单词、数字、标点和换行符，词表只由训练语料构建，能够保持较小输出层规模。

### 4.2.2 GPT-2 tokenizer

说明 GPT-2 tokenizer 使用成熟的 BPE 编码，能处理更多文本形式，但固定词表较大，会显著增加小模型输出层参数量。

### 4.2.3 分词策略选择

说明本实验最终选择正则分词作为主要方案，因为它比字符级分词更容易生成完整单词，同时比 GPT-2 tokenizer 更适合小语料和低算力实验。

## 4.3 预训练任务与训练策略

本实验使用 next-token prediction 作为预训练目标。

### 4.3.1 滑动窗口样本构造

说明将 token 序列按照 context_length 和 stride 划分为多个训练样本，输入为当前窗口，目标为整体右移一位后的 token 序列。

### 4.3.2 数据集划分方式

说明本实验支持 token split 和 sample split。对于较小本地语料，sample split 可以避免验证集 token 数不足的问题；对于较大数据集，可以使用训练集和验证集分开构造数据。

### 4.3.3 训练设置

说明 batch size、learning rate、weight decay、epochs、eval frequency 等主要训练参数。

### 4.3.4 模型保存与早停策略

说明训练过程中保存 last checkpoint，可选保存 best checkpoint，并根据验证集 loss 使用 early stopping 控制训练过程。

### 4.3.5 文本生成策略

说明预训练后使用 prompt 进行自回归生成，并支持 temperature、top-k、repetition penalty 和 no-repeat n-gram 等采样控制方法。

## 4.4 预训练结果与生成效果分析

本节展示预训练实验结果。

### 4.4.1 Loss 与 Perplexity 曲线

展示 train loss、val loss、train perplexity 和 val perplexity，并说明 loss 下降趋势。

### 4.4.2 模型保存结果

说明输出目录中保存了模型权重、词表配置、训练参数、metrics、loss 曲线、日志和生成样例。

### 4.4.3 文本生成样例

展示不同 prompt 下的生成结果，并分析小模型生成效果。需要说明：由于模型规模和语料规模有限，生成结果不要求达到真实大模型水平，重点是验证预训练流程有效。

---

# 5 微调实验与任务适应

## 5.1 微调任务与数据构建

说明本实验选择一个简单监督任务，例如情感二分类、垃圾短信分类或简单问答任务。为了保持项目简洁，建议选择文本分类任务。

### 5.1.1 任务说明

说明分类任务的输入、输出和类别设置。

### 5.1.2 监督数据格式

说明微调数据使用 CSV 或 JSON 格式，包含文本和标签字段。

## 5.2 微调方法设计

说明在预训练 MiniGPT 基础上添加分类头，使用最后一个 token 或平均池化后的隐藏状态作为文本表示。

### 5.2.1 模型加载

说明加载预训练 checkpoint 和 tokenizer 配置，保证微调阶段与预训练阶段使用相同词表。

### 5.2.2 分类头设计

说明分类头将 GPT 输出的隐藏状态映射为类别概率。

### 5.2.3 微调训练流程

说明读取监督数据、编码文本、构造 batch、计算交叉熵损失、更新模型参数并保存微调结果。

## 5.3 微调结果与分析

展示微调过程和结果。

### 5.3.1 微调 loss 与 accuracy

展示训练 loss、验证 loss 和 accuracy。

### 5.3.2 测试样例

列出若干输入文本、真实标签和模型预测结果。

### 5.3.3 微调前后对比

说明预训练模型主要学习语言模式，而微调后模型能够适应具体任务。

---

# 6 总结与体会

## 6.1 项目完成情况

总结本项目完成了小型 GPT 模型搭建、Tokenizer 设计、预训练程序、文本生成测试和微调任务。

## 6.2 实验问题与改进方向

简要说明实验中遇到的问题，例如小语料导致生成效果有限、正则 tokenizer 泛化能力不如 BPE、CPU 训练速度较慢、模型参数量有限等。

后续可以改进为：扩大训练语料、使用更成熟的 BPE tokenizer、增加模型规模、引入学习率调度和梯度裁剪、尝试指令微调等。

## 6.3 学习收获与体会

总结通过本项目对 GPT 模型结构、分词、预训练、微调和推理生成流程的理解。

---

# 参考文献与开源资源说明

1. Sebastian Raschka, Build a Large Language Model From Scratch, Manning, 2024.
2. LLMs-from-scratch GitHub 项目。
3. PyTorch 官方文档。
4. Attention Is All You Need.
5. Hugging Face Datasets 文档。
6. TinyStories 数据集说明。
7. tiktoken 分词工具相关资料。

---

# 附录

## 附录 A：项目文件结构

```text
my_tiny_gpt/
├── mini_gpt.py
├── tiny_tokenizer.py
├── train_pretrain.py
├── generate.py
├── finetune_classifier.py
├── train_pretrain.ps1
├── data/
│   ├── pretrain.txt
│   └── finetune.csv
└── outputs/
    └── 训练输出目录/
```

## 附录 B：主要运行命令

```bash
python my_tiny_gpt/mini_gpt.py
python my_tiny_gpt/tiny_tokenizer.py
python my_tiny_gpt/train_pretrain.py
python my_tiny_gpt/generate.py
python my_tiny_gpt/finetune_classifier.py
```

## 附录 C：主要输出文件

```text
vocab.json
args.json
metrics.json
pretrain_loss.csv
pretrain_loss.png
pretrain_log.txt
pretrain_sample.txt
pretrain_model_last.pt
pretrain_model_best.pt
```

## 附录 D：实验结果截图清单

1. 模型结构测试输出；
2. tokenizer 分词测试输出；
3. 预训练运行过程截图；
4. loss 曲线图；
5. metrics.json 关键结果；
6. 文本生成样例；
7. 微调训练结果；
8. GitHub 项目文件结构截图。
