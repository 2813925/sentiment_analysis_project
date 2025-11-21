# BERT 中文评论情感分析微调项目 (SFT + DPO)

## 项目结构
```
sentiment_analysis_project/
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据
│   ├── processed/                 # 处理后的数据
│   └── dpo/                       # DPO训练数据
├── models/                        # 模型保存目录
│   ├── bert_baseline/             # 基线模型
│   ├── bert_sft/                  # SFT微调模型
│   └── bert_dpo/                  # DPO微调模型
├── scripts/                       # 训练脚本
│   ├── data_preparation.py        # 数据准备
│   ├── train_baseline.py          # 基线训练
│   ├── train_sft.py               # SFT训练
│   ├── train_dpo.py               # DPO训练
│   └── evaluate.py                # 评估脚本
├── demo/                          # Demo应用
│   └── gradio_app.py              # Gradio演示界面
├── utils/                         # 工具函数
│   ├── data_loader.py             # 数据加载
│   └── metrics.py                 # 评估指标
├── notebooks/                     # Jupyter notebooks
│   └── analysis.ipynb             # 数据分析
├── requirements.txt               # 依赖包
└── README.md                      # 说明文档
```

## 环境要求
- Python 3.8+
- CUDA 11.0+ (GPU训练)
- 至少16GB内存

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备数据
```bash
python scripts/1_data_preparation.py
```

### 3. 训练模型
```bash
# 训练基线模型
python scripts/2_baseline_training.py

# SFT微调
python scripts/3_sft_training.py

# DPO微调
python scripts/4_dpo_training.py
```

### 4. 评估模型
```bash
python scripts/5_evaluation.py
```

### 5. 启动Demo
```bash
python demo/6_demo_app.py
```

## 项目亮点
1. ✅ 完整的SFT + DPO训练流程
2. ✅ 多模型对比（BERT baseline vs SFT vs DPO）
3. ✅ 自构指令数据和偏好数据
4. ✅ 可视化训练过程
5. ✅ Gradio交互式Demo
6. ✅ 详细的评估指标
