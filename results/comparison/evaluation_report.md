# 模型评估报告

生成时间: 2025-11-20 00:37:21

## 模型性能对比

| Model | Accuracy | Precision | Recall | F1 Score | Macro F1 |
| --- | --- | --- | --- | --- | --- |
| BERT Baseline | 0.8995 | 0.9042 | 0.8995 | 0.9008 | 0.8866 |
| BERT + SFT | 0.9046 | 0.9067 | 0.9046 | 0.9053 | 0.8911 |
| BERT + SFT + DPO | 0.8827 | 0.9043 | 0.8827 | 0.8860 | 0.8728 |


## 详细分类报告

### BERT Baseline

```
              precision    recall  f1-score   support

          负面     0.8074    0.8934    0.8482       244
          正面     0.9486    0.9023    0.9249       532

    accuracy                         0.8995       776
   macro avg     0.8780    0.8978    0.8866       776
weighted avg     0.9042    0.8995    0.9008       776

```

### BERT + SFT

```
              precision    recall  f1-score   support

          负面     0.8295    0.8770    0.8526       244
          正面     0.9421    0.9173    0.9295       532

    accuracy                         0.9046       776
   macro avg     0.8858    0.8972    0.8911       776
weighted avg     0.9067    0.9046    0.9053       776

```

### BERT + SFT + DPO

```
              precision    recall  f1-score   support

          负面     0.7429    0.9590    0.8372       244
          正面     0.9783    0.8477    0.9084       532

    accuracy                         0.8827       776
   macro avg     0.8606    0.9034    0.8728       776
weighted avg     0.9043    0.8827    0.8860       776

```

## 结论

- **最佳模型**: BERT + SFT
- **F1分数**: 0.9053
- **准确率**: 0.9046

- **相对改进**: -1.64%
  - Baseline F1: 0.9008
  - Final F1: 0.8860
