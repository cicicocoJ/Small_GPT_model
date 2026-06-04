# 基于 PyTorch 的小型 GPT 模型构建与训练

## 摘要

本文围绕“从零构建大模型”课程任务，基于 PyTorch 设计并实现了一个小参数量 GPT 类语言模型。项目主要包括大语言模型关键知识点梳理、小型 GPT 模型架构搭建、TinyStories 数据集预训练、文本生成测试以及特定任务微调等内容。实验采用适合低算力平台的小型模型配置，通过正则分词或 GPT-2 BPE 分词将文本转换为 token 序列，并使用 next-token prediction 作为预训练目标。实验结果表明，小型 GPT 模型虽然无法达到真实大语言模型的生成质量，但能够较完整地体现文本编码、词嵌入、位置编码、因果自注意力、预训练和微调等核心机制。

**关键词：** GPT；Transformer；PyTorch；TinyStories；预训练；微调；Tokenizer

---

# 目录

* [1 绪论](#1-绪论)

  * [1.1 项目背景与意义](#11-项目背景与意义)
  * [1.2 项目任务与技术路线](#12-项目任务与技术路线)

* [2 大语言模型相关理论基础与关键知识点](#2-大语言模型相关理论基础与关键知识点)

  * [2.1 GPT 类语言模型基本原理](#21-gpt-类语言模型基本原理)
  * [2.2 文本分词与输入表示](#22-文本分词与输入表示)
  * [2.3 Transformer 核心结构](#23-transformer-核心结构)
  * [2.4 预训练与微调机制](#24-预训练与微调机制)

* [3 小型 GPT 模型设计与实现](#3-小型-gpt-模型设计与实现)

  * [3.1 模型总体结构](#31-模型总体结构)
  * [3.2 核心模块设计](#32-核心模块设计)
  * [3.3 模型参数配置与低算力适配](#33-模型参数配置与低算力适配)

* [4 训练数据准备与模型预训练实验](#4-训练数据准备与模型预训练实验)

  * [4.1 TinyStories 数据集准备](#41-tinystories-数据集准备)
  * [4.2 文本拼接、分词与样本构造](#42-文本拼接分词与样本构造)
  * [4.3 预训练任务与训练策略](#43-预训练任务与训练策略)
  * [4.4 预训练结果、推理生成与效果分析](#44-预训练结果推理生成与效果分析)

* [5 微调实验与任务适应](#5-微调实验与任务适应)

  * [5.1 微调任务与监督数据构建](#51-微调任务与监督数据构建)
  * [5.2 微调策略与程序设计](#52-微调策略与程序设计)
  * [5.3 微调结果与任务效果分析](#53-微调结果与任务效果分析)

* [6 总结与体会](#6-总结与体会)

  * [6.1 项目完成情况与任务要求对应](#61-项目完成情况与任务要求对应)
  * [6.2 实验问题与改进方向](#62-实验问题与改进方向)
  * [6.3 学习收获与体会](#63-学习收获与体会)

* [参考文献与开源资源说明](#参考文献与开源资源说明)

* [附录](#附录)

---

# 1 绪论

## 1.1 项目背景与意义

简要介绍大语言模型和 GPT 类模型的发展背景，说明 GPT 模型在文本生成、自然语言处理和人工智能应用中的代表性。

说明本项目不是训练真正的大规模商用模型，而是在低算力条件下实现一个小型 GPT 模型，通过完整流程理解大语言模型的基本原理。

## 1.2 项目任务与技术路线

概括课程作业要求，包括知识点梳理、模型搭建、预训练、微调、推理展示和总结体会。

技术路线建议用五个步骤概括：数据准备与分词编码、训练样本构造、小型 GPT 模型搭建、预训练与生成测试、微调与结果分析。

---

# 2 大语言模型相关理论基础与关键知识点

## 2.1 GPT 类语言模型基本原理

介绍大语言模型的基本概念，说明 GPT 属于 Decoder-only Transformer 结构，是一种自回归语言模型。

**自回归语言建模。** GPT 类模型根据已有上下文预测下一个 token，训练目标是 next-token prediction。

**GPT 的基本计算流程。** 文本经过 tokenizer 转换为 token id，再经过 token embedding、position embedding、多层 Transformer Block 和线性输出层，得到对下一个 token 的概率预测。

## 2.2 文本分词与输入表示

说明自然语言文本不能直接输入神经网络，需要先转换为 token 序列，再映射为连续向量表示。

**Tokenizer 的作用。** Tokenizer 完成文本切分、词表构建、编码和解码。

**正则分词方法。** 本实验主要采用基于正则表达式的 tokenizer，将文本切分为英文单词、数字、标点和换行符等 token。该方法相比字符级分词更容易保留完整单词，同时词表规模小于 GPT-2 BPE，更适合小模型训练。

**GPT-2 BPE 分词方法。** 代码中保留 GPT-2 BPE tokenizer 作为可选方案。该方法更接近真实 GPT 模型中的子词分词方式，但固定词表较大，会增加输出层参数量。

**词嵌入与位置嵌入。** Token embedding 表示 token 本身，position embedding 表示 token 所处位置，两者相加后作为 Transformer 的输入。

## 2.3 Transformer 核心结构

说明 Transformer 是 GPT 类模型的核心结构，本实验重点实现其中的因果多头自注意力、前馈网络、残差连接和层归一化。

**因果多头自注意力。** 自注意力用于建模上下文关系，因果 mask 用于屏蔽未来 token，保证模型只能根据前文进行预测。

**前馈网络、残差连接与层归一化。** 前馈网络用于非线性变换，残差连接和 LayerNorm 用于稳定深层网络训练。

## 2.4 预训练与微调机制

说明预训练和微调是大语言模型训练流程中的两个重要阶段。

**预训练。** 使用无标签文本，通过 next-token prediction 学习通用语言模式。

**微调。** 在预训练模型基础上，使用带标签数据或任务格式数据继续训练，使模型适应特定任务。

**二者区别。** 预训练侧重学习通用语言规律，微调侧重学习具体任务目标。

---

# 3 小型 GPT 模型设计与实现

## 3.1 模型总体结构

说明本实验实现的小型 GPT 模型采用 Decoder-only Transformer 架构，主要由 token embedding、position embedding、多层 Transformer Block、最终 LayerNorm 和线性输出层组成。

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

本节介绍 `mini_gpt.py` 中的主要模块。由于这些内容对应实际代码结构，可以保留少量三级标题。

### 3.2.1 MiniGPTConfig

说明模型配置类中包含 vocab_size、context_length、emb_dim、n_heads、n_layers、dropout 等参数。

### 3.2.2 CausalSelfAttention

说明该模块实现因果多头自注意力，通过上三角 mask 屏蔽未来位置。

### 3.2.3 TransformerBlock

说明每个 Transformer Block 由 LayerNorm、因果自注意力、前馈网络和残差连接组成。

### 3.2.4 MiniGPT 模型主体

说明 `MiniGPT` 由 embedding 层、Transformer Block、最终 LayerNorm 和输出层组成，输出形状为 `[batch_size, seq_len, vocab_size]`，用于计算 next-token prediction loss。

## 3.3 模型参数配置与低算力适配

说明本实验为了适应普通电脑或 CPU/GPU 单机环境，采用较小参数规模，并通过减小层数、嵌入维度、上下文长度和 batch size 控制训练成本。

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

# 4 训练数据准备与模型预训练实验

## 4.1 TinyStories 数据集准备

本实验统一使用 Hugging Face 上的 TinyStories 数据集作为预训练语料。该数据集由大量简短英文故事构成，语言结构相对简单，适合小参数模型学习基本英文生成模式。

### 4.1.1 数据集加载与划分

说明训练脚本通过 Hugging Face datasets 加载 TinyStories，并通过 `hf_train_split` 和 `hf_val_split` 指定训练集和验证集切片。

### 4.1.2 数据缓存与本地保存

说明程序设置 Hugging Face 缓存目录，并支持将数据集保存到本地磁盘，方便重复实验和离线加载。

## 4.2 文本拼接、分词与样本构造

说明从 TinyStories 中提取故事文本，将多条文本拼接为连续语料，并使用 `<|endoftext|>` 分隔不同故事。随后对文本进行清洗、分词和编码。

### 4.2.1 文本清洗与拼接

说明文本清洗包括统一换行、去除多余空行等操作，拼接时使用特殊分隔符区分不同故事。

### 4.2.2 Tokenizer 选择与编码

说明本实验主要采用正则 tokenizer，并保留 GPT-2 BPE tokenizer 作为可选方案。分词后将文本转换为 token id 序列。

### 4.2.3 滑动窗口样本构造

说明将 token 序列按照 `context_length` 和 `stride` 划分为多个训练样本，输入为当前窗口，目标为整体右移一位后的 token 序列。

## 4.3 预训练任务与训练策略

本实验使用 next-token prediction 作为预训练目标，使用交叉熵损失函数和 AdamW 优化器训练模型。

### 4.3.1 训练参数设置

说明 batch size、learning rate、weight decay、epochs、eval frequency、early stopping patience 等主要训练参数。

### 4.3.2 模型保存与日志记录

说明训练过程中保存 last checkpoint，可选保存 best checkpoint，同时保存 args、metrics、loss 曲线和训练日志。

### 4.3.3 文本生成策略

说明预训练后使用 prompt 进行自回归文本生成，并支持 temperature、top-k、repetition penalty 和 no-repeat n-gram 等采样控制方法。

## 4.4 预训练结果、推理生成与效果分析

本节展示预训练实验结果。

### 4.4.1 Loss 与 Perplexity 曲线

展示 train loss、val loss、train perplexity 和 val perplexity，并说明 loss 下降趋势。

### 4.4.2 文本生成样例

展示不同 prompt 下的生成结果，并分析小模型生成效果。需要说明生成结果不要求达到真实大模型水平，重点是验证预训练流程有效。

### 4.4.3 预训练效果分析

结合 loss 曲线、perplexity 和生成样例，分析模型是否学习到基本文本模式，并说明小模型和小规模训练的局限性。

---

# 第5章 微调实验与任务适应

## 5.1 微调任务设计

本节说明为什么在预训练完成后需要进行监督微调，以及本实验选择的具体下游任务。预训练阶段的 MiniGPT 主要完成 next-token prediction，只能根据上下文预测下一个 token；而微调阶段则将模型适配到一个明确的监督任务中，用于验证预训练模型的任务迁移能力。

本实验的微调任务不再采用普通情感分类，而是设计为 **故事结局倾向二分类任务**（Story Ending Polarity Classification）。模型输入为一段简短英文故事结尾，输出为 `positive` 或 `negative`。其中，`positive` 表示故事结局积极，例如问题得到解决、角色得到帮助、冲突缓和、人物安全或开心；`negative` 表示故事结局消极，例如问题没有解决、角色失败、伤心、害怕、孤单，或坏结果持续存在。该任务与预训练语料 TinyStories 的儿童故事风格保持一致，比普通影评式情感分类更适合本项目。

本节可重点写清楚：任务输入是什么、输出类别是什么、为什么该任务能体现“任务适应”、为什么它和 TinyStories 预训练语料同域。

## 5.2 监督数据构建

本节说明微调数据集的来源、标注方式和数据格式。由于预训练阶段使用的是 Hugging Face 上的 TinyStories 数据集，微调阶段也从 TinyStories 风格文本中抽取候选故事结尾，以减少预训练语料和下游任务之间的分布差异。原始 TinyStories 文本并不包含分类标签，因此本文对抽取出的候选样本进行了人工标注，将其转化为监督分类数据。

微调数据保存为 CSV 文件，主要字段包括 `text` 和 `label`。其中 `text` 为故事结尾文本，`label` 为对应类别，取值为 `positive` 或 `negative`。标注时只依据文本中呈现的信息判断结局倾向：如果困难最终得到解决，则标为 `positive`；如果困难没有解决或坏结果持续存在，则标为 `negative`；对于结局不明确或难以判断的样本则剔除。最终用于训练的数据集为 `finetune_300_labeled_detailed.csv`，划分为训练集、验证集和测试集，数量分别为 210、45 和 45。

本节可补充说明数据预处理流程：读取 CSV 后，复用预训练阶段保存的 tokenizer 对文本编码；根据模型的 `context_length=64` 进行截断与 padding；同时生成 `attention_mask`，用于区分真实 token 和 padding token。

## 5.3 微调模型与程序设计

本节说明微调程序如何在预训练 MiniGPT 的基础上构建分类模型。微调阶段首先加载预训练得到的 `pretrain_model_best.pt`，并复用同一输出目录下的 `vocab.json`，从而保证预训练和微调阶段使用相同词表、相同 token 编码方式。这样可以避免重新训练 tokenizer 导致词表不一致。

在模型结构上，本文没有修改 MiniGPT 主体，而是在 MiniGPT 后增加一个线性分类头。输入文本经过 token embedding、position embedding、多层 Transformer block 和 final norm 后，得到每个 token 的隐藏状态。由于分类任务需要得到整段文本的表示，本文采用平均池化方式，将所有有效 token 的隐藏状态聚合为文本级向量，再输入线性分类头，输出 `negative` 和 `positive` 两个类别的 logits。随后使用 softmax 得到类别概率，并通过交叉熵损失进行监督训练。

本节还可以说明程序输出内容：训练过程中保存 `finetune_history.csv`、loss 曲线、accuracy 曲线；训练完成后保存 `metrics.json`、`classification_report.txt`、`finetune_predictions.csv` 和 `finetune_errors.csv`，用于后续结果分析。

## 5.4 微调策略与训练设置

本节说明最终采用的微调策略以及为什么这样选择。本文曾参考文本分类微调中的部分参数微调方法，尝试冻结大部分预训练模型参数，仅训练最后一个 Transformer block、final norm 和分类头。但由于本文构建的 MiniGPT 只有 2 层 Transformer，模型规模较小，过度冻结会限制模型对下游任务的适应能力。因此，最终主实验采用 **全参数微调**（full fine-tuning），即 MiniGPT 主体和分类头全部参与训练。

文本表示方式上，本文选择 mean pooling 作为主实验方案。与只使用最后一个 token 表示相比，平均池化可以综合整段故事结尾中的多个 token 信息，更适合故事结局倾向判断这种整体语义分类任务。训练时使用 AdamW 优化器，学习率为 `5e-5`，权重衰减为 `0.01`，batch size 为 8，训练轮数上限为 60，并设置 early stopping patience 为 30。训练过程中以验证集表现选择最佳 checkpoint，最终测试时使用的是验证集最优模型，而不是最后一轮模型。

本节可以强调：本实验不是单纯追求训练集准确率，而是通过验证集选择最优模型，以减少训练后期过拟合对最终测试结果的影响。

## 5.5 微调结果与任务效果分析

本节展示微调实验结果，包括训练过程、验证集表现、测试集指标和样例预测。最终主实验在第 39 轮取得最佳验证集表现，验证集准确率为 82.22%，验证损失为 0.5188。测试集上，模型准确率达到 84.44%，macro F1 为 80.62%，weighted F1 为 83.87%。

从训练过程看，随着训练进行，训练损失整体下降，训练准确率逐步提高，说明模型能够从监督数据中学习故事结局分类特征。但第 39 轮之后，训练集准确率继续提高，而验证集准确率和验证损失出现波动，说明后期存在一定过拟合趋势。由于最终测试采用的是验证集最优 checkpoint，因此过拟合对最终测试结果的影响得到了控制。

本节还应展示混淆矩阵和分类报告。测试集中，模型对 positive 类识别更稳定，positive 类 F1 为 89.23%；negative 类 F1 为 72.00%，说明模型对负向结局的识别能力相对较弱。混淆矩阵中，真实 negative 被误判为 positive 的样本有 5 条，真实 positive 被误判为 negative 的样本有 2 条。

在样例分析中，可以列出若干预测正确和预测错误的文本。正确样例可以展示模型能够识别“回家安全”“得到帮助”“朋友和好”等积极结局，也能识别“角色受伤”“坏结果持续”等消极结局。错误样例重点分析“负面事件 + 学到教训”这类边界样本，例如文本中出现 `learned to be careful` 时，模型可能将“学到教训”误判为积极结局，而忽略前面角色失去朋友、伤心等负面结果。

## 5.6 本章小结

本章完成了从预训练 MiniGPT 到下游分类任务的微调实验。实验首先构建了 TinyStories 同域的故事结局倾向分类数据集，然后在预训练模型基础上增加分类头，并采用全参数微调和 mean pooling 的方式完成监督训练。实验结果表明，微调后的 MiniGPT 能够较好地区分积极和消极故事结局，测试准确率达到 84.44%，说明预训练语言模型不仅可以用于文本生成，也可以通过监督微调适应特定分类任务。

同时，实验也暴露出小型模型和小规模数据集的局限性：模型对 positive 类识别更稳定，对 negative 类尤其是“负面事件中包含道德教训”的边界样本仍容易误判；训练后期也存在一定过拟合趋势。后续可以继续扩充负向样本、改进数据平衡性，或尝试更多微调策略，以进一步提升模型的泛化能力。


# 6 总结与体会

## 6.1 项目完成情况与任务要求对应

从知识点梳理、模型搭建、预训练、微调和总结体会五个方面，对照课程要求说明本项目的完成情况。

可以使用表格形式：

| 作业要求   | 报告对应章节 | 完成内容                          |
| ------ | ------ | ----------------------------- |
| 知识点梳理  | 第 2 章  | GPT 原理、分词、Transformer、预训练与微调  |
| 模型架构搭建 | 第 3 章  | 小型 GPT 模型、核心模块、低算力配置          |
| 预训练    | 第 4 章  | TinyStories 数据、分词、样本构造、预训练与生成 |
| 微调     | 第 5 章  | 监督数据、微调策略、微调结果                |
| 总结体会   | 第 6 章  | 项目完成情况、问题与收获                  |

## 6.2 实验问题与改进方向

简要说明实验中遇到的问题，例如小模型生成质量有限、正则 tokenizer 泛化能力不如 BPE、CPU 训练速度较慢、模型参数量有限等。

后续可以改进为：扩大训练数据规模、尝试 GPT-2 BPE 或更成熟的 tokenizer、增加模型规模、引入学习率调度和梯度裁剪、尝试指令微调等。

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
│   ├── hf_cache/
│   ├── TinyStories_train/
│   ├── TinyStories_val/
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
3. TinyStories 数据加载与预训练运行过程截图；
4. loss 曲线图；
5. metrics.json 关键结果；
6. 文本生成样例；
7. 微调训练结果；
8. GitHub 项目文件结构截图。
