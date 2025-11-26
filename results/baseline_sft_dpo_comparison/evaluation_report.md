# Baseline vs SFT vs DPO 评估对比报告

**生成时间**: 2025年11月26日 19:08:43

---

## 📋 模型信息

### BERT Baseline
- **路径**: `/opt/data/private/zxw/sentiment/models/bert_baseline/final`
- **类型**: standard
- **描述**: 标准BERT分类器

### SFT Optimized
- **路径**: `/opt/data/private/zxw/sentiment/models/bert_sft_optimized`
- **类型**: standard
- **描述**: SFT优化模型 (F1≈98%)

### DPO Best
- **路径**: `/opt/data/private/zxw/sentiment/models/bert_dpo_best`
- **类型**: mlp
- **描述**: DPO最佳模型 (F1=89.47%)

---

## 📊 性能对比

| 模型 Model | 准确率 Accuracy | 精确率 Precision | 召回率 Recall | F1分数 F1 Score | ROC-AUC | AP |
|---------|----------|-----------|--------|----------|---------|----|
| **BERT Baseline** | 0.8103 | 0.7640 | 0.8500 | **0.8047** | 0.9116 | 0.9126 |
| **SFT Optimized** | 0.7989 | 0.7473 | 0.8500 | **0.7953** | 0.9106 | 0.9101 |
| **DPO Best** | 0.9080 | 0.9444 | 0.8500 | **0.8947** | 0.9771 | 0.9730 |

---

## 📈 详细分类报告

### BERT Baseline

```
              precision    recall  f1-score   support

          负面     0.8588    0.7766    0.8156        94
          正面     0.7640    0.8500    0.8047        80

    accuracy                         0.8103       174
   macro avg     0.8114    0.8133    0.8102       174
weighted avg     0.8152    0.8103    0.8106       174

```

### SFT Optimized

```
              precision    recall  f1-score   support

          负面     0.8554    0.7553    0.8023        94
          正面     0.7473    0.8500    0.7953        80

    accuracy                         0.7989       174
   macro avg     0.8013    0.8027    0.7988       174
weighted avg     0.8057    0.7989    0.7991       174

```

### DPO Best

```
              precision    recall  f1-score   support

          负面     0.8824    0.9574    0.9184        94
          正面     0.9444    0.8500    0.8947        80

    accuracy                         0.9080       174
   macro avg     0.9134    0.9037    0.9066       174
weighted avg     0.9109    0.9080    0.9075       174

```

## 🎯 混淆矩阵

### BERT Baseline

```
              预测 Predicted
           负面      正面
实际 负面     73        21
     正面     12        68
```

### SFT Optimized

```
              预测 Predicted
           负面      正面
实际 负面     71        23
     正面     12        68
```

### DPO Best

```
              预测 Predicted
           负面      正面
实际 负面     90         4
     正面     12        68
```

---

## 🔍 关键发现

### 🏆 最佳模型

- **模型**: DPO Best
- **F1 Score**: 0.8947 (89.47%)
- **Accuracy**: 0.9080 (90.80%)
- **ROC-AUC**: 0.9771

### 📊 性能排名 (按F1 Score)

1. 🥇 **DPO Best**: F1 = 0.8947 (89.47%)
2. 🥈 **BERT Baseline**: F1 = 0.8047 (80.47%)
3. 🥉 **SFT Optimized**: F1 = 0.7953 (79.53%)

### 📉 模型对比分析

**Baseline → SFT**:
- Baseline F1: 0.8047 (80.47%)
- SFT F1: 0.7953 (79.53%)
- 提升: -0.0094 (-1.17%)

**SFT → DPO**:
- SFT F1: 0.7953 (79.53%)
- DPO F1: 0.8947 (89.47%)
- 变化: +0.0994 (+12.50%)
- ✅ DPO进一步提升了性能

**Baseline → DPO (总体)**:
- Baseline F1: 0.8047
- DPO F1: 0.8947
- 总提升: +0.0900 (+11.18%)

---

## 💡 结论

### 实验总结

1. **最佳模型**: DPO Best (F1=0.8947)
2. **稳定性**: 所有模型都展现出良好的稳定性
3. **泛化能力**: ROC-AUC和AP指标表明模型泛化良好

### 💻 生产建议

- **部署模型**: 推荐使用 **DPO Best**
- **原因**: DPO模型通过偏好学习进一步优化了性能

### 🔮 优化方向

- **数据增强**: 增加训练数据量，提高数据质量
- **架构改进**: 尝试更大的模型或不同的架构
- **超参数调优**: 继续优化学习率、dropout等参数
- **集成学习**: 结合多个模型的优势

---

**报告结束**
