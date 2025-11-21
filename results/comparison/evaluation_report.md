# 模型评估报告

生成时间: 2025-11-16 21:37:11

## 模型性能对比

| Model | Accuracy | Precision | Recall | F1 Score | Macro F1 |
| --- | --- | --- | --- | --- | --- |
| BERT Baseline | 0.8943 | 0.9059 | 0.8943 | 0.8966 | 0.8832 |
| BERT + SFT | 0.9188 | 0.9187 | 0.9188 | 0.9188 | 0.9057 |
| BERT + SFT + DPO | 0.9137 | 0.9199 | 0.9137 | 0.9150 | 0.9032 |


## 详细分类报告

### BERT Baseline

```
              precision    recall  f1-score   support

          负面     0.7774    0.9303    0.8470       244
          正面     0.9649    0.8778    0.9193       532

    accuracy                         0.8943       776
   macro avg     0.8711    0.9041    0.8832       776
weighted avg     0.9059    0.8943    0.8966       776

```

### BERT + SFT

```
              precision    recall  f1-score   support

          负面     0.8724    0.8689    0.8706       244
          正面     0.9400    0.9417    0.9408       532

    accuracy                         0.9188       776
   macro avg     0.9062    0.9053    0.9057       776
weighted avg     0.9187    0.9188    0.9188       776

```

### BERT + SFT + DPO

```
              precision    recall  f1-score   support

          负面     0.8195    0.9303    0.8714       244
          正面     0.9659    0.9060    0.9350       532

    accuracy                         0.9137       776
   macro avg     0.8927    0.9182    0.9032       776
weighted avg     0.9199    0.9137    0.9150       776

```

## 结论

- **最佳模型**: BERT + SFT
- **F1分数**: 0.9188
- **准确率**: 0.9188

- **相对改进**: 2.06%
  - Baseline F1: 0.8966
  - Final F1: 0.9150
