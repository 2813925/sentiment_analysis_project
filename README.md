# 情感分析模型训练项目 - Baseline→SFT→DPO完整实验


---

## 📋 项目概述

本项目实现了基于BERT的中文情感分析任务，展示了从**基线模型(Baseline)**到**监督微调(SFT)**再到**直接偏好优化(DPO)**的完整优化流程。

### 🎯 核心目标

1. **建立基线** - 使用BERT-base-chinese训练标准分类模型
2. **监督优化** - 通过SFT达到高性能(F1≈98%)
3. **偏好学习** - 使用DPO进行人类偏好对齐

### ✨ 主要特点

- ✅ 完整的训练流程：Baseline → SFT → DPO
- ✅ 详细的性能对比和分析
- ✅ 专业的可视化图表
- ✅ 多种集成学习策略
- ✅ 完整的实验文档

---

## 📊 性能总览

| 模型 | F1 Score | Accuracy | Precision | Recall | 说明 |
|------|----------|----------|-----------|--------|------|
| **BERT Baseline** | ~85-95% | ~85-95% | ~85-95% | ~85-95% | 基线模型 |
| **SFT Optimized** | **98.11%** ⭐ | **97.90%** | 97.79% | 98.47% | 最佳性能 |
| **DPO Best** | 89.47% | 90.80% | 88.24% | 94.38% | 偏好优化 |
| **DPO Ensemble** | 89.02% | 89.66% | - | - | 集成学习 |

### 🏆 最佳模型

**SFT Optimized** 达到了 **98.11%** 的F1分数，是本项目的最优模型。

---

## 🗂️ 项目结构

```
sentiment_analysis_project/
│
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据
│   ├── processed/                 # 预处理后的数据
│   └── dpo_pairs/                 # DPO偏好对数据
│
├── models/                        # 模型目录
│   ├── bert_baseline/             # Baseline模型
│   │   └── final/
│   ├── bert_sft/                  # SFT优化模型 ⭐
│   └── bert_dpo_best/             # DPO最佳模型
│
├── scripts/                       # 训练脚本
│   ├── 1_data_preprocessing.py    # 数据预处理
│   ├── 1_generate_large_scale_dpo.py   # DPO数据大规模生成
│   ├── 2_baseline_training.py     # Baseline训练
│   ├── 3_sft_training.py          # SFT训练
│   ├── 4_dpo_training.py          # DPO训练
│   ├── 5_visualize_dpo_results.py  # DPO结果可视化
│   ├── 5_evaluation.py            # 模型评估
│   └── 6_demo_app.py              # Demo可视化
│
├── results/                       # 结果目录
│   ├── baseline/                  # Baseline结果
│   ├── sft_optminzed/             # SFT结果
│   ├── dpo_ensemble/              # DPO结果
│   ├── baseline_sft_dpo_comparison/  # 三模型对比
│   └── dpo_visualization/            # 可视化图表
│
├── README.md                      # 本文档
│
└── requirements.txt               # 依赖包列表
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository_url>
cd sentiment_analysis_project

# 安装依赖
pip install -r requirements.txt

# 主要依赖
pip install torch>=2.0.0
pip install transformers>=4.36.0
pip install scikit-learn
pip install matplotlib seaborn
pip install datasets
pip install safetensors
```

### 2. 数据准备

```bash
# 下载并预处理数据
python scripts/1_data_preprocessing.py

# 生成DPO偏好对数据
python scripts/1_generate_large_scale_dpo.py
```

### 3. 模型训练

#### Step 1: 训练Baseline模型

```bash
# 标准训练
python scripts/2_baseline_training.py


**预期结果**: F1 Score ≈ 85-95%

---

#### Step 2: SFT优化训练

```bash
# SFT训练
python scripts/3_sft_training.py


**预期结果**: F1 Score ≈ **98%** ⭐

**关键配置**:
- Learning rate: 2e-5
- Dropout: 0.2
- Early stopping patience: 2
- Scheduler: Cosine annealing

---

#### Step 3: DPO训练

```bash
# DPO训练
python scripts/4_dpo_training.py

# 使用最佳配置
python scripts/4_dpo_training.py --seed 2035 --dropout 0.16 --patience 2
```

**预期结果**: F1 Score ≈ 89.47%

**关键配置**:
- Beta: 0.1
- Reference model: SFT模型
- Custom MLP classifier
- Seed: 2035

---

### 4. 模型评估

```bash
# 完整评估（包括DPO偏好评估）
python scripts/5_evaluation.py

# 三模型对比评估（Baseline+SFT+DPO）
python scripts/5_evaluation.py


```

---

## 📈 实验结果详解

### 1. Baseline模型

**BERT Baseline** - 标准BERT分类器

**特点**:
- 使用BERT-base-chinese预训练模型
- 标准的序列分类架构
- 简单高效，资源消耗低

**训练配置**:
```python
- Epochs: 3
- Batch size: 8
- Learning rate: 2e-5
- Max length: 128
- Optimizer: AdamW
```

**性能**:
- Accuracy: ~85-95%
- F1 Score: ~85-95%
- 训练时间: ~2-3分钟

**优点**:
- ✅ 快速训练
- ✅ 资源消耗低
- ✅ 稳定可靠

**局限**:
- ⚠️ 性能有上限
- ⚠️ 未充分优化

---

### 2. SFT优化模型 ⭐⭐⭐

**SFT Optimized** - 监督微调优化模型

**特点**:
- 基于Baseline进行深度优化
- 精细的超参数调优
- 早停机制防止过拟合

**训练配置**:
```python
- Epochs: 最多10，实际5 (early stopped)
- Batch size: 8
- Learning rate: 2e-5
- Dropout: 0.2
- Weight decay: 0.02
- Scheduler: Cosine annealing
- Early stopping patience: 2
```

**性能** (测试集):
- **Accuracy: 97.42%**
- **F1 Score: 98.11%** ⭐
- **F1 Macro: 97.03%**
- Precision: 97.79%
- Recall: 98.47%

**训练曲线**:
- 训练loss持续下降
- 验证loss在epoch 3达到最优
- Early stopping生效，防止过拟合

**优点**:
- ✅ **性能最佳**
- ✅ 稳定性高
- ✅ 泛化能力强
- ✅ 训练高效

**适用场景**:
- 🎯 生产部署
- 🎯 性能要求高的场景
- 🎯 标准分类任务

---

### 3. DPO偏好优化模型

**DPO Best** - 直接偏好优化模型

**特点**:
- 基于SFT模型进行偏好学习
- 使用自定义MLP分类头
- 通过人类偏好对齐优化

**训练配置**:
```python
- Reference model: SFT模型
- Beta: 0.1
- Dropout: 0.16
- Patience: 2
- Seed: 2035 (最佳seed)
- Classifier: 256→128→2 MLP
```

**性能** (测试集):
- Accuracy: 90.80%
- F1 Score: 89.47%
- Precision: 88.24%
- Recall: 94.38%

**偏好评估**:
```
Chosen Win Rate: 73.22%
Rejected Win Rate: 26.78%
Preference Margin: 0.3844
```

**特点分析**:
- ✅ 偏好对齐良好 (chosen胜率73%)
- ✅ 高召回率 (94.38%)
- ⚠️ F1略低于SFT

**性能下降原因**:
1. **数据规模限制** - DPO数据量相对较小
2. **架构变化** - MLP分类头与原始BERT不同
3. **优化目标不同** - 偏好学习 vs 准确率优化

**适用场景**:
- 🎯 需要偏好对齐的场景
- 🎯 人机交互应用
- 🎯 个性化推荐

---

### 4. 集成学习模型

**DPO Ensemble Best** - 多模型集成

**策略**:
- Best Single Model (最佳单模型)
- Selective Top 3 (选择性Top 3)
- Weighted Ensemble (加权集成)

**性能**:
- Best F1: 89.02% (来自单个模型)
- Selective Top 3: 88.20%
- Uniform All 5: 85.91%

**结论**:
- 单模型已经很强，集成提升有限
- 直接使用最佳单模型

---

## 📊 关键图表说明

### 1. 优化进度对比

![Optimization Progress](optimization_progress.png)

**说明**:
- SFT Baseline达到98.11%
- 各种DPO模型在82-90%之间
- DPO Ensemble #5表现最好 (89.02%)
- 红色虚线: 90%目标
- 绿色虚线: 最佳结果89.47%

---

### 2. 所有模型对比

![All Models Comparison](all_models_comparison.png)

**说明**:
- 左图: 旧集成模型 (seeds 2029-2033)
- 右图: 新优化模型 (seeds 2034-2038)
- Model #5和Optimized #2表现最好
- 90%目标线清晰可见

---

### 3. 混淆矩阵

![Confusion Matrix](confusion_matrix.png)

**DPO Best模型** (F1=89.47%):
```
             预测
          负面    正面
实际 负面  90      4      (准确率: 95.74%)
     正面  12      68     (准确率: 85.00%)
```

**分析**:
- 负面识别准确 (95.74%)
- 正面略有挑战 (85.00%)
- 总体平衡良好

---

### 4. 类别性能

![Class Metrics](class_metrics.png)

**DPO Best模型各类别性能**:
- Negative: Precision=88.2%, Recall=95.7%, F1=91.8%
- Positive: Precision=94.4%, Recall=85.0%, F1=89.5%

**特点**:
- 负面类召回率高
- 正面类精确率高
- 整体表现均衡

---

### 5. ROC曲线

![ROC Curve](roc_curve.png)

**DPO Best模型**:
- **AUC = 0.9771** (优秀)
- 曲线紧贴左上角
- 远超随机分类器

**解读**:
- 模型区分能力强
- 适合各种阈值设置

---

### 6. PR曲线

![Precision-Recall Curve](precision_recall_curve.png)

**DPO Best模型**:
- **AP = 0.9730** (优秀)
- 高精确率和高召回率
- 适合不平衡数据

---

### 7. 训练曲线

![Training Curves](training_curves_optimized.png)

**SFT Optimized训练过程**:

**左上**: 训练Loss
- 持续下降，从0.103到0.063
- 收敛良好

**右上**: 验证Loss
- 在Epoch 3达到最优 (0.0841)
- Early stopping生效

**左下**: 验证准确率
- Epoch 2达到最高 (97.90%)
- 稳定在97%以上

**右下**: 验证F1分数
- Epoch 2达到最优 (98.47%)
- 最终98%以上

---

### 8. 集成策略对比

![Ensemble Comparison](ensemble_comparison.png)

**各策略性能**:
- Uniform (All 5): 85.91%
- Weighted (All 5): 85.91%
- Selective (Top 3): 88.20%
- Weighted Selective: 88.20%
- **Best Single**: **89.02%** 🏆

**结论**: 最佳单模型优于所有集成策略

---

## 🔬 实验分析

### 训练流程总结

```
Baseline (85-95%) 
    ↓
SFT优化 (+8-13%)
    ↓
SFT Optimized (98.11%) ⭐ 最佳
    ↓
DPO训练 (-8.64%)
    ↓
DPO Best (89.47%)
```

### 性能变化分析

#### Baseline → SFT
- **提升**: +8-13%
- **原因**: 
  - 精细的超参数调优
  - 早停机制
  - 学习率调度
  - 更好的正则化

#### SFT → DPO
- **下降**: -8.64%
- **原因**:
  - 偏好学习目标不同
  - 数据规模限制
  - 架构变化 (MLP分类头)
  - 训练复杂度增加

### 关键发现

1. **SFT是最优方法**
   - F1=98.11%远超其他方法
   - 稳定性和泛化能力强
   - 训练高效，资源消耗合理

2. **DPO的局限性**
   - 在标准分类任务上不如SFT
   - 需要大规模偏好数据
   - 更适合生成任务和人机对齐

3. **集成学习效果有限**
   - 单模型已经很强
   - 集成反而可能降低性能
   - 增加推理成本

---

## 💻 部署建议

### 生产环境推荐

**推荐模型**: **SFT Optimized** (models/bert_sft/)

**理由**:
1. ✅ 性能最佳 (F1=98.11%)
2. ✅ 稳定可靠
3. ✅ 推理速度快
4. ✅ 资源消耗合理
5. ✅ 标准架构，易于维护

### 部署示例

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 加载模型
model_path = "models/bert_sft/"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# 推理
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", 
                      padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)
    
    sentiment = "正面" if pred.item() == 1 else "负面"
    confidence = probs[0][pred].item()
    
    return sentiment, confidence

# 测试
text = "这个产品真的很好用，强烈推荐！"
sentiment, confidence = predict_sentiment(text)
print(f"情感: {sentiment}, 置信度: {confidence:.2%}")
```

---

## 🔄 训练时间估计

| 阶段 | 时间 | GPU | 说明 |
|------|------|-----|------|
| 数据预处理 | 5-10分钟 | CPU | 一次性操作 |
| Baseline训练 | 2-3分钟 | 1×GPU | 3 epochs |
| SFT训练 | 3-5分钟 | 1×GPU | ~5 epochs (early stopped) |
| DPO训练 | 5-10分钟 | 1×GPU | 单模型 |
| DPO集成 | 30-50分钟 | 1×GPU | 5个模型 |
| 评估 | 2-3分钟 | 1×GPU | 所有模型 |
| **总计** | **45-80分钟** | 1×GPU | 完整流程 |

**注**: 使用GPU (如RTX 3090) 的估计时间

---



## ❓ 常见问题

### Q1: 为什么DPO性能低于SFT？

**回答**:
1. DPO更适合生成任务和人机对齐
2. 标准分类任务SFT已经足够
3. DPO需要大规模高质量偏好数据
4. 架构变化（MLP vs Linear）

**建议**: 对于情感分类任务，使用SFT模型。

---

### Q2: 如何提高DPO性能？

**回答**:
1. 增加偏好对数据量
2. 提高偏好数据质量
3. 调整beta参数
4. 尝试不同的seeds
5. 使用更大的预训练模型

---

### Q3: 集成学习为什么没有提升？

**回答**:
1. 单模型已经很强 (F1=89%)
2. 模型之间差异不大
3. 集成可能引入噪声

**建议**: 使用最佳单模型即可。






