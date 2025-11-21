"""
SFT (Supervised Fine-Tuning) 训练脚本 - Step 3
使用指令格式对BERT进行微调
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt


class SFTTrainer:
    def __init__(self, base_dir: str = "./"):
        # 使用绝对路径，防止工作目录变化导致找不到文件
        self.base_dir = Path(base_dir).resolve()
        self.data_dir = self.base_dir / "data" / "processed"
        self.model_dir = self.base_dir / "models" / "bert_sft"
        self.results_dir = self.base_dir / "results" / "sft"

        # 创建目录
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  使用设备: {self.device}")
        print(f"📁 项目根目录: {self.base_dir}")

        # 模型参数：改为本地模型路径
        # 使用原始预训练 BERT 作为 SFT 起点
        self.model_name = str(self.base_dir / "models" / "bert-base-chinese")

        # 如果你想基于 baseline 继续训练，可以改成这一行：
        # self.model_name = str(self.base_dir / "models" / "bert_baseline" / "final")

        self.max_length = 256  # SFT需要更长的上下文
        self.num_labels = 2

        # 强制只用本地缓存，禁止联网
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    def load_sft_data(self):
        """加载SFT格式数据"""
        print("\n📊 加载SFT数据...")

        with open(self.data_dir / "sft_train.json", 'r', encoding='utf-8') as f:
            train_data = json.load(f)

        with open(self.data_dir / "sft_dev.json", 'r', encoding='utf-8') as f:
            dev_data = json.load(f)

        with open(self.data_dir / "sft_test.json", 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        print(f"训练集: {len(train_data)} | 验证集: {len(dev_data)} | 测试集: {len(test_data)}")

        return train_data, dev_data, test_data

    def format_instruction(self, item):
        """格式化指令式输入"""
        # item 结构示例：
        # {
        #   "instruction": "请判断以下酒店评论的情感倾向，回答'正面'或'负面'。",
        #   "input": "房间很干净，就是有点小。",
        #   "output": "正面"
        # }
        full_text = f"{item['instruction']}\n{item['input']}\n答案："
        return full_text

    def label_to_id(self, label_text: str) -> int:
        """将 '正面' / '负面' 转成 1 / 0"""
        # 保险起见，做一下 strip
        label_text = str(label_text).strip()
        if label_text == "正面":
            return 1
        elif label_text == "负面":
            return 0
        # 如果不符合预期，打印一下，默认当作负面
        print(f"⚠️ 意外标签值: {label_text}，默认映射为 0（负面）")
        return 0

    def prepare_dataset(self, data, tokenizer):
        """准备数据集"""

        texts = []
        labels = []

        for item in data:
            # 构造指令式文本
            text = self.format_instruction(item)
            texts.append(text)

            # 从 output 字段读取标签（正面/负面）
            label_id = self.label_to_id(item["output"])
            labels.append(label_id)

        # Tokenize
        encodings = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 创建Dataset
        dataset_dict = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels)
        }

        dataset = Dataset.from_dict(dataset_dict)

        return dataset

    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        f1_macro = f1_score(labels, predictions, average='macro')

        return {
            'accuracy': acc,
            'f1': f1,
            'f1_macro': f1_macro
        }

    def train(self):
        """训练SFT模型"""
        print("\n" + "=" * 60)
        print("🚀 开始SFT训练")
        print("=" * 60)

        # 加载数据
        train_data, dev_data, test_data = self.load_sft_data()

        # 加载tokenizer和模型（只用本地文件）
        print(f"\n📦 加载模型: {self.model_name}")
        tokenizer = BertTokenizer.from_pretrained(
            self.model_name,
            local_files_only=True,  # 关键：只用本地
        )
        model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            local_files_only=True,  # 关键：只用本地
        )

        # 准备数据集
        print("\n🔧 准备SFT数据集...")
        train_dataset = self.prepare_dataset(train_data, tokenizer)
        eval_dataset = self.prepare_dataset(dev_data, tokenizer)
        test_dataset = self.prepare_dataset(test_data, tokenizer)

        print("训练样本示例:")
        print(f"  输入: {self.format_instruction(train_data[0])[:100]}...")
        print(f"  标签(output): {train_data[0]['output']}")

        # 训练参数
        training_args = TrainingArguments(
            output_dir=str(self.model_dir),
            num_train_epochs=4,  # SFT可以多训练几轮
            per_device_train_batch_size=16,  # 序列更长，batch size小一些
            per_device_eval_batch_size=16,
            learning_rate=3e-5,  # 稍高的学习率
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_dir=str(self.results_dir / "logs"),
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=300,
            save_strategy="steps",
            save_steps=300,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="tensorboard",
            fp16=torch.cuda.is_available(),
        )

        # 创建Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        # 开始训练
        print("\n🏋️ 开始SFT训练...")
        train_result = trainer.train()

        # 保存模型
        print(f"\n💾 保存SFT模型到: {self.model_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(self.model_dir)

        # 评估
        print("\n📊 评估SFT模型...")

        # 验证集评估
        eval_results = trainer.evaluate(eval_dataset)
        print("\n验证集结果:")
        for key, value in eval_results.items():
            try:
                print(f"  {key}: {value:.4f}")
            except TypeError:
                print(f"  {key}: {value}")

        # 测试集评估
        test_results = trainer.evaluate(test_dataset)
        print("\n测试集结果:")
        for key, value in test_results.items():
            try:
                print(f"  {key}: {value:.4f}")
            except TypeError:
                print(f"  {key}: {value}")

        # 详细分类报告
        predictions = trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = [self.label_to_id(item["output"]) for item in test_data]

        report = classification_report(
            true_labels,
            pred_labels,
            target_names=['负面', '正面'],
            digits=4
        )
        print("\n分类报告:")
        print(report)

        # 保存结果
        results = {
            'model_name': 'BERT + SFT',
            'train_samples': len(train_data),
            'eval_samples': len(dev_data),
            'test_samples': len(test_data),
            'eval_results': eval_results,
            'test_results': test_results,
            'classification_report': report,
            'training_time': train_result.metrics.get('train_runtime', None),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(self.results_dir / "results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n✅ 结果已保存到: {self.results_dir / 'results.json'}")

        # 绘制训练曲线
        self.plot_training_curves(trainer)

        # 测试几个例子
        self.test_examples(model, tokenizer, test_data[:5])

        return trainer, results

    def test_examples(self, model, tokenizer, examples):
        """测试一些例子"""
        print("\n🧪 测试示例:")

        model.eval()
        model.to(self.device)

        for i, item in enumerate(examples):
            text = self.format_instruction(item)

            inputs = tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()

            pred_label = '正面' if pred == 1 else '负面'
            true_label = item["output"]

            # SFT 数据里没有 review 字段，用 input 当作原始评论
            review_text = item.get("input", "")[:50]

            print(f"\n示例 {i + 1}:")
            print(f"  评论: {review_text}...")
            print(f"  真实标签(output): {true_label}")
            print(f"  预测标签: {pred_label}")
            print("  ✓" if pred_label == true_label else "  ✗")

    def plot_training_curves(self, trainer):
        """绘制训练曲线"""
        print("\n📈 绘制训练曲线...")

        log_history = trainer.state.log_history

        # 提取训练损失和评估指标
        train_loss = [log['loss'] for log in log_history if 'loss' in log]
        eval_loss = [log['eval_loss'] for log in log_history if 'eval_loss' in log]
        eval_f1 = [log['eval_f1'] for log in log_history if 'eval_f1' in log]

        steps_train = [log['step'] for log in log_history if 'loss' in log]
        steps_eval = [log['step'] for log in log_history if 'eval_loss' in log]

        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss曲线
        axes[0].plot(steps_train, train_loss, label='Train Loss', marker='o', alpha=0.7)
        axes[0].plot(steps_eval, eval_loss, label='Eval Loss', marker='s', alpha=0.7)
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('SFT Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # F1曲线
        axes[1].plot(steps_eval, eval_f1, label='Eval F1', marker='s', alpha=0.7)
        axes[1].set_xlabel('Steps')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('SFT Validation F1 Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'sft_training_curves.png', dpi=300, bbox_inches='tight')
        print(f"✅ 训练曲线已保存到: {self.results_dir / 'sft_training_curves.png'}")

        plt.close()


if __name__ == "__main__":
    trainer = SFTTrainer()
    trainer.train()

    print("\n" + "=" * 60)
    print("✅ SFT训练完成！")
    print("=" * 60)
    print("\n📌 下一步:")
    print("  运行: python scripts/4_dpo_training.py")
