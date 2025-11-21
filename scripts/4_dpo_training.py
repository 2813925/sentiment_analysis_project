"""
DPO (Direct Preference Optimization) 训练脚本 - Step 4
使用偏好对数据进行DPO微调（简化版）
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt


class DPOTrainer:
    def __init__(self, base_dir: str = "./"):
        # 使用绝对路径，防止工作目录变化导致找不到文件
        self.base_dir = Path(base_dir).resolve()
        self.data_dir = self.base_dir / "data"
        self.sft_model_dir = self.base_dir / "models" / "bert_sft"
        self.model_dir = self.base_dir / "models" / "bert_dpo"
        self.results_dir = self.base_dir / "results" / "dpo"

        # 创建目录
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  使用设备: {self.device}")
        print(f"📁 项目根目录: {self.base_dir}")

        # DPO参数
        self.beta = 0.1  # DPO温度参数（当前简化版里主要保留以备扩展）
        self.max_length = 256
        self.num_labels = 2

        # 强制只用本地缓存，禁止联网
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    def load_dpo_data(self):
        """加载DPO偏好对数据"""
        print("\n📊 加载DPO偏好对数据...")

        dpo_file = self.data_dir / "dpo_pairs" / "dpo_train.json"

        if not dpo_file.exists():
            print(f"❌ DPO数据不存在: {dpo_file}")
            print("请先运行: python scripts/1_data_preparation.py")
            return None

        with open(dpo_file, 'r', encoding='utf-8') as f:
            dpo_data = json.load(f)

        print(f"DPO训练对数: {len(dpo_data)}")

        # 划分训练集和验证集
        split_idx = int(0.9 * len(dpo_data))
        train_data = dpo_data[:split_idx]
        eval_data = dpo_data[split_idx:]

        print(f"训练集: {len(train_data)} | 验证集: {len(eval_data)}")

        return train_data, eval_data

    def label_text_to_id(self, label_text: str) -> int:
        """将标签文本映射到 0/1，支持英文和中文"""
        if label_text is None:
            return 0
        t = str(label_text).strip()
        if t in ["positive", "正面"]:
            return 1
        if t in ["negative", "负面"]:
            return 0

        print(f"⚠️ 意外标签值: {t}，默认映射为 0（负面）")
        return 0

    def compute_dpo_loss(self, model, ref_model, batch):
        """
        原始 DPO 损失（当前简化版脚本里没有直接用到，保留作为参考）
        """
        # 获取输入
        prompt_input_ids = batch['prompt_input_ids'].to(self.device)
        prompt_attention_mask = batch['prompt_attention_mask'].to(self.device)

        chosen_input_ids = batch['chosen_input_ids'].to(self.device)
        chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
        chosen_labels = batch['chosen_labels'].to(self.device)

        rejected_input_ids = batch['rejected_input_ids'].to(self.device)
        rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
        rejected_labels = batch['rejected_labels'].to(self.device)

        # 当前策略模型的输出
        chosen_outputs = model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask,
            labels=chosen_labels
        )
        rejected_outputs = model(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask,
            labels=rejected_labels
        )

        # 参考模型的输出（不需要梯度）
        with torch.no_grad():
            ref_chosen_outputs = ref_model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask,
                labels=chosen_labels
            )
            ref_rejected_outputs = ref_model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask,
                labels=rejected_labels
            )

        # 计算对数概率
        chosen_logps = -chosen_outputs.loss
        rejected_logps = -rejected_outputs.loss
        ref_chosen_logps = -ref_chosen_outputs.loss
        ref_rejected_logps = -ref_rejected_outputs.loss

        # DPO损失
        pi_logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        loss = -F.logsigmoid(self.beta * (pi_logratios - ref_logratios)).mean()

        return loss

    def simplified_dpo_train(self):
        """简化版DPO训练（基于分类任务的改进）"""
        print("\n" + "=" * 60)
        print("🚀 开始DPO训练（简化版）")
        print("=" * 60)

        # 加载数据
        result = self.load_dpo_data()
        if result is None:
            return

        train_data, eval_data = result

        # 加载SFT模型作为起点
        print(f"\n📦 加载SFT模型: {self.sft_model_dir}")

        if not self.sft_model_dir.exists():
            print("⚠️  SFT模型不存在，将使用baseline模型")
            # 优先用 baseline 的 final 目录
            fallback_dir = self.base_dir / "models" / "bert_baseline" / "final"
            if fallback_dir.exists():
                self.sft_model_dir = fallback_dir
            else:
                # 再退一步用 bert-base-chinese 预训练模型
                self.sft_model_dir = self.base_dir / "models" / "bert-base-chinese"

        tokenizer = BertTokenizer.from_pretrained(
            str(self.sft_model_dir),
            local_files_only=True,
        )
        model = BertForSequenceClassification.from_pretrained(
            str(self.sft_model_dir),
            num_labels=self.num_labels,
            local_files_only=True,
        )

        # 准备训练数据：使用 prompt 作为输入，chosen 作为“偏好标签”
        print("\n🔧 准备DPO训练数据...")

        train_texts = []
        train_labels = []

        for item in train_data:
            # 使用 prompt 作为输入，可以加上 “答案：” 保持和 SFT 一致
            text = item["prompt"] + "\n答案："
            train_texts.append(text)

            # chosen 是人工认为正确的情感标签，例如 “正面” / “负面”
            label_id = self.label_text_to_id(item.get("chosen"))
            train_labels.append(label_id)

        # Tokenize
        train_encodings = tokenizer(
            train_texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': torch.tensor(train_labels)
        })

        # 验证集：同样用 prompt + chosen
        eval_texts = []
        eval_labels = []

        for item in eval_data:
            text = item["prompt"] + "\n答案："
            eval_texts.append(text)
            label_id = self.label_text_to_id(item.get("chosen"))
            eval_labels.append(label_id)

        eval_encodings = tokenizer(
            eval_texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        eval_dataset = Dataset.from_dict({
            'input_ids': eval_encodings['input_ids'],
            'attention_mask': eval_encodings['attention_mask'],
            'labels': torch.tensor(eval_labels)
        })

        # 训练参数：数据很少，用较小学习率 + 较少 epoch，避免把 SFT 训练“带坏”
        training_args = TrainingArguments(
            output_dir=str(self.model_dir),
            num_train_epochs=2,              # epoch 少一点
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=5e-6,             # 比 SFT 更小的学习率
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_dir=str(self.results_dir / "logs"),
            logging_steps=20,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="tensorboard",
            fp16=torch.cuda.is_available(),
        )

        def compute_metrics(eval_pred):
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

        # 创建Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        # 开始训练
        print("\n🏋️ 开始DPO微调...")
        train_result = trainer.train()

        # 保存模型
        print(f"\n💾 保存DPO模型到: {self.model_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(self.model_dir)

        # 评估（验证集）
        print("\n📊 评估DPO模型（验证集）...")
        eval_results = trainer.evaluate(eval_dataset)

        print("\n验证集结果:")
        for key, value in eval_results.items():
            try:
                print(f"  {key}: {value:.4f}")
            except TypeError:
                print(f"  {key}: {value}")

        # 在测试集上评估：复用 SFT 的测试数据（指令格式）
        test_file = self.data_dir / "processed" / "sft_test.json"
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        test_texts = [f"{item['instruction']}\n{item['input']}\n答案：" for item in test_data]
        # SFT 测试数据中，标签在 output 字段，值为 “正面” / “负面”
        test_labels = [self.label_text_to_id(item.get("output")) for item in test_data]

        test_encodings = tokenizer(
            test_texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        test_dataset = Dataset.from_dict({
            'input_ids': test_encodings['input_ids'],
            'attention_mask': test_encodings['attention_mask'],
            'labels': torch.tensor(test_labels)
        })

        print("\n📊 评估DPO模型（测试集）...")
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

        report = classification_report(
            test_labels,
            pred_labels,
            target_names=['负面', '正面'],
            digits=4
        )
        print("\n分类报告:")
        print(report)

        # 保存结果
        results = {
            'model_name': 'BERT + SFT + DPO',
            'dpo_train_samples': len(train_data),
            'dpo_eval_samples': len(eval_data),
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

        return trainer, results


if __name__ == "__main__":
    trainer = DPOTrainer()
    trainer.simplified_dpo_train()

    print("\n" + "=" * 60)
    print("✅ DPO训练完成！")
    print("=" * 60)
    print("\n📌 下一步:")
    print("  运行: python scripts/5_evaluation.py")
