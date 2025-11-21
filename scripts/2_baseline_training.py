#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT Baseline 训练脚本 - 使用本地模型
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_from_disk
import matplotlib
matplotlib.rc("font",family='YouYuan')

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# 创建必要的目录
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class BERTBaselineTrainer:
    """BERT Baseline 训练器"""

    def __init__(self, model_name="bert-base-chinese", use_quick_mode=False, use_local=False):
        """
        初始化训练器

        Args:
            model_name: 预训练模型名称
            use_quick_mode: 是否使用快速模式（用于测试）
            use_local: 是否使用本地模型
        """
        self.use_local = use_local

        # 如果使用本地模型，指向本地路径
        if use_local:
            local_model_path = MODELS_DIR / model_name
            if not local_model_path.exists():
                raise FileNotFoundError(
                    f"本地模型不存在: {local_model_path}\n"
                    f"请确保已下载模型文件到该目录"
                )
            self.model_name = str(local_model_path)
            print(f"📁 使用本地模型: {local_model_path}")
        else:
            self.model_name = model_name
            # 设置国内镜像源
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            print(f"🌐 使用在线模型: {model_name} (镜像源: {os.environ['HF_ENDPOINT']})")

        self.use_quick_mode = use_quick_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 输出目录
        self.output_dir = MODELS_DIR / "bert_baseline"
        self.results_dir = RESULTS_DIR / "baseline"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        print(f"🖥️  使用设备: {self.device}")
        print(f"🔧 快速模式: {'开启' if use_quick_mode else '关闭'}")

    def load_data(self):
        """加载预处理后的数据"""
        print("\n📊 加载数据...")

        processed_dir = DATA_DIR / "processed"
        self.train_dataset = load_from_disk(str(processed_dir / "train"))
        self.val_dataset = load_from_disk(str(processed_dir / "validation"))
        self.test_dataset = load_from_disk(str(processed_dir / "test"))

        # 快速模式：只使用少量数据
        if self.use_quick_mode:
            self.train_dataset = self.train_dataset.select(range(min(500, len(self.train_dataset))))
            self.val_dataset = self.val_dataset.select(range(min(100, len(self.val_dataset))))
            self.test_dataset = self.test_dataset.select(range(min(100, len(self.test_dataset))))
            print("⚡ 快速模式：使用少量数据进行训练")

        print(f"训练集: {len(self.train_dataset)} | "
              f"验证集: {len(self.val_dataset)} | "
              f"测试集: {len(self.test_dataset)}")

    def train(self):
        """训练BERT模型"""
        print("\n" + "=" * 60)
        print("🚀 开始训练BERT Baseline模型")
        print("=" * 60)

        # 加载数据
        self.load_data()

        # 加载tokenizer和模型
        print(f"\n📦 加载模型: {self.model_name}")
        try:
            tokenizer = BertTokenizer.from_pretrained(self.model_name)
            model = BertForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2,
                use_safetensors=True  # 支持 safetensors 格式
            ).to(self.device)
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            print("\n💡 建议：")
            print("1. 如果是网络问题，请使用 --local 参数并手动下载模型")
            print("2. 手动下载地址: https://hf-mirror.com/google-bert/bert-base-chinese")
            print("3. 下载后放到: models/bert-base-chinese/ 目录")
            raise

        # Tokenize数据
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=128
            )

        print("\n🔄 Tokenizing数据...")
        train_dataset = self.train_dataset.map(tokenize_function, batched=True)
        val_dataset = self.val_dataset.map(tokenize_function, batched=True)
        test_dataset = self.test_dataset.map(tokenize_function, batched=True)

        # 设置训练参数
        if self.use_quick_mode:
            num_epochs = 1
            batch_size = 8  # 减小batch size避免OOM
            save_steps = 100
        else:
            num_epochs = 3
            batch_size = 8  # 减小batch size避免OOM
            save_steps = 500

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=save_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            report_to="none",
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=2,  # 梯度累积，减少显存占用
            dataloader_pin_memory=False,  # 减少显存占用
        )

        # 定义评估指标
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)

            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='binary'
            )

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        # 创建Trainer
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )

        # 开始训练
        print("\n🏋️  开始训练...")
        train_result = trainer.train()

        # 保存模型
        print("\n💾 保存模型...")
        trainer.save_model(str(self.output_dir / "final"))
        tokenizer.save_pretrained(str(self.output_dir / "final"))

        # 保存训练指标
        metrics = train_result.metrics
        with open(self.results_dir / "train_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"✅ 模型已保存到: {self.output_dir / 'final'}")

        return trainer

    def evaluate(self, trainer):
        """在测试集上评估模型"""
        print("\n" + "=" * 60)
        print("📊 在测试集上评估模型")
        print("=" * 60)

        # Tokenize测试集（如果还没有tokenize）
        if 'input_ids' not in self.test_dataset.column_names:
            print("🔄 Tokenizing测试集...")
            tokenizer = trainer.tokenizer

            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=128
                )

            test_dataset_tokenized = self.test_dataset.map(tokenize_function, batched=True)
        else:
            test_dataset_tokenized = self.test_dataset

        # 评估
        metrics = trainer.evaluate(test_dataset_tokenized)

        # 保存评估指标
        with open(self.results_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # 打印结果
        print("\n测试集结果:")
        print(f"  Accuracy:  {metrics['eval_accuracy']:.4f}")
        print(f"  Precision: {metrics['eval_precision']:.4f}")
        print(f"  Recall:    {metrics['eval_recall']:.4f}")
        print(f"  F1 Score:  {metrics['eval_f1']:.4f}")

        # 生成详细报告
        self._generate_detailed_report(trainer)

        return metrics

    def _generate_detailed_report(self, trainer):
        """生成详细的评估报告"""
        print("\n📝 生成详细评估报告...")

        # 获取预测结果
        predictions = trainer.predict(self.test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids

        # 分类报告
        report = classification_report(
            true_labels,
            pred_labels,
            target_names=["Negative", "Positive"],
            digits=4
        )

        # 保存报告
        report_path = self.results_dir / "classification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("BERT Baseline 分类报告\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)

        print(f"✅ 分类报告已保存到: {report_path}")

        # 绘制混淆矩阵
        self._plot_confusion_matrix(true_labels, pred_labels)

    def _plot_confusion_matrix(self, true_labels, pred_labels):
        """绘制混淆矩阵"""
        cm = confusion_matrix(true_labels, pred_labels)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=["负面", "正面"],
            yticklabels=["负面", "正面"]
        )
        plt.title('BERT Baseline - 混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')

        # 保存图片
        cm_path = self.results_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 混淆矩阵已保存到: {cm_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="训练BERT Baseline模型")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速模式（用于测试）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-chinese",
        help="预训练模型名称"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="使用本地模型（需要手动下载）"
    )

    args = parser.parse_args()

    # 创建训练器
    trainer_obj = BERTBaselineTrainer(
        model_name=args.model,
        use_quick_mode=args.quick,
        use_local=args.local
    )

    # 训练模型
    trainer = trainer_obj.train()

    # 评估模型
    metrics = trainer_obj.evaluate(trainer)

    print("\n" + "=" * 60)
    print("✅ BERT Baseline训练完成！")
    print("=" * 60)
    print(f"\n最终测试集准确率: {metrics['eval_accuracy']:.4f}")
    print(f"模型保存位置: {trainer_obj.output_dir / 'final'}")
    print(f"结果保存位置: {trainer_obj.results_dir}")


if __name__ == "__main__":
    main()
