# 从零构建小型 GPT 作业项目

本目录用于完成课程作业《从零构建大模型》。参考项目为 `LLMs-from-scratch-main`，个人实现代码放在 `my_tiny_gpt/` 中。

## 当前目标

本项目不追求训练真正的大模型，而是在低算力环境下完成一个小型 GPT 类模型实验，覆盖：

1. 文本数据处理
2. GPT 模型架构搭建
3. 小规模预训练
4. 微调
5. 推理生成
6. 报告素材整理

## 目录说明

- `LLMs-from-scratch-main/`: 课程推荐参考项目，主要参考 ch04、ch05、ch06。
- `my_tiny_gpt/`: 本作业的个人实验代码。
- `my_tiny_gpt/data/`: 存放训练语料和后续微调数据。
- `my_tiny_gpt/outputs/`: 保存词表、模型权重、loss 记录、生成样例等实验结果。
- `my_tiny_gpt/report_assets/`: 保存报告中需要使用的图片、表格和文本素材。
- `报告/`: 课程报告相关文件。

## 已完成工作

1. 阅读并梳理 `LLMs-from-scratch-main` 的 README 和 ch02 到 ch06 主线代码。
2. 创建 `my_tiny_gpt/` 项目结构。
3. 实现 `my_tiny_gpt/mini_gpt.py`：
   - token embedding
   - position embedding
   - causal self-attention
   - multi-head attention
   - feed forward
   - layer norm
   - residual connection
   - 前向传播测试
4. 实现 `my_tiny_gpt/tiny_tokenizer.py`：
   - GPT-2 BPE tokenizer
   - `encode`
   - `decode`
   - 保存和加载 GPT-2 tokenizer 配置 `outputs/vocab.json`
5. 实现 `my_tiny_gpt/train_pretrain.py`：
   - 读取 `data/pretrain.txt`
   - 构建滑动窗口预训练数据集
   - 训练 MiniGPT 做 next-token prediction
   - 保存模型、loss、日志和生成样例

## 推荐工作流程

### 1. 构建词表

```powershell
python .\my_tiny_gpt\tiny_tokenizer.py
```

输出：

- `my_tiny_gpt/outputs/vocab.json`

### 2. 测试小型 GPT 前向传播

```powershell
python .\my_tiny_gpt\mini_gpt.py
```

预期结果：输入 token ids 后输出形状为 `[batch_size, seq_len, vocab_size]` 的 logits。

### 3. 运行小规模预训练

快速测试：

```powershell
python .\my_tiny_gpt\train_pretrain.py --epochs 1 --eval_freq 1 --cpu
```

正式一点的小实验：

```powershell
python .\my_tiny_gpt\train_pretrain.py --epochs 20 --cpu
```

输出：

- `my_tiny_gpt/outputs/pretrain_model.pt`
- `my_tiny_gpt/outputs/pretrain_loss.csv`
- `my_tiny_gpt/outputs/pretrain_loss.png`
- `my_tiny_gpt/outputs/pretrain_log.txt`
- `my_tiny_gpt/outputs/pretrain_sample.txt`

## 后续计划

1. 实现 `my_tiny_gpt/generate.py`，加载预训练模型并生成文本。
2. 准备 `my_tiny_gpt/data/finetune.csv`，构建一个简单分类微调任务。
3. 实现 `my_tiny_gpt/finetune_classifier.py`，完成监督微调并保存准确率结果。
4. 实现或整理 `my_tiny_gpt/run_all.py`，一键运行主要实验流程。
5. 将 loss 曲线、生成样例、微调结果整理到报告中。

## 报告对应关系

- 大模型相关知识点：参考 `LLMs-from-scratch-main/README.md`、ch04、ch05、ch06，并结合个人实现说明。
- 小型 GPT 架构搭建：对应 `my_tiny_gpt/mini_gpt.py`。
- 预训练过程：对应 `my_tiny_gpt/tiny_tokenizer.py` 和 `my_tiny_gpt/train_pretrain.py`。
- 微调过程：后续对应 `my_tiny_gpt/finetune_classifier.py`。
- 成果总结和体会：结合 `outputs/` 中的实验结果编写。


cd my_tiny_gpt
python tiny_tokenizer.py
python mini_gpt.py

python train_pretrain.py --epochs 20 --cpu
.\train_pretrain.ps1