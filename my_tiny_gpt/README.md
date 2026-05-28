# my_tiny_gpt

本目录用于课程作业《从零构建大模型》的个人实验实现。

目标是构建一个小型 GPT 类模型，完整覆盖：

1. 文本数据处理
2. GPT 模型架构搭建
3. 小规模预训练
4. 分类微调
5. 文本生成推理
6. 报告素材整理

当前目录只建立项目骨架，后续逐步补充代码。

## 目录说明

- `data/`: 存放小规模预训练文本和微调数据。
- `outputs/`: 保存训练曲线、模型权重、生成样例和微调结果。
- `report_assets/`: 保存报告中使用的图片、表格和文本素材。
- `tiny_tokenizer.py`: 分词和词表构建。
- `tiny_gpt.py`: 小型 GPT 模型结构。
- `train_pretrain.py`: 预训练脚本。
- `generate.py`: 文本生成脚本。
- `finetune_classifier.py`: 分类微调脚本。
- `run_all.py`: 一键运行主要实验流程。
