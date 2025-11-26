"""
综合评估脚本 - Step 5
对比所有模型的性能
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 只用本地模型


class ModelEvaluator:
    def __init__(self, base_dir: str = "./"):
        # 使用绝对路径，避免工作目录变化
        self.base_dir = Path(base_dir).resolve()
        self.data_dir = self.base_dir / "data" / "processed"
        self.models_dir = self.base_dir / "models"
        self.results_dir = self.base_dir / "results" / "comparison"

        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  使用设备: {self.device}")

        # 要评估的模型路径（baseline 用最终的 final 目录）
        self.models = {
            "BERT Baseline": self.models_dir / "bert_baseline" / "final",
            "BERT + SFT": self.models_dir / "bert_sft",
            "BERT + SFT + DPO": self.models_dir / "bert_dpo",
        }

        self.max_length = 256

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

    def load_test_data(self):
        """加载测试数据"""
        print("\n📊 加载测试数据...")

        with open(self.data_dir / "sft_test.json", "r", encoding="utf-8") as f:
            test_data = json.load(f)

        print(f"测试集样本数: {len(test_data)}")

        return test_data

    def evaluate_model(self, model_name, model_path, test_data):
        """评估单个模型"""
        print(f"\n🔍 评估模型: {model_name}")

        if not model_path.exists():
            print(f"  ⚠️  模型不存在: {model_path}")
            return None

        # 加载模型（本地）
        tokenizer = BertTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        model = BertForSequenceClassification.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        model.to(self.device)
        model.eval()

        # 准备测试数据：指令 + 输入 + “答案：”
        texts = [f"{item['instruction']}\n{item['input']}\n答案：" for item in test_data]

        # 标签：优先用 output（正面/负面），兼容可能存在的 sentiment 字段
        true_labels = [
            self.label_text_to_id(
                item.get("output", item.get("sentiment"))
            )
            for item in test_data
        ]

        # 批量预测
        predictions = []
        batch_size = 32

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                inputs = tokenizer(
                    batch_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(preds)

        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        # 计算指标
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average="weighted")
        recall = recall_score(true_labels, predictions, average="weighted")
        f1 = f1_score(true_labels, predictions, average="weighted")
        f1_macro = f1_score(true_labels, predictions, average="macro")

        # 混淆矩阵
        cm = confusion_matrix(true_labels, predictions)

        # 分类报告（这里用中文标签名）
        report = classification_report(
            true_labels,
            predictions,
            target_names=["负面", "正面"],
            digits=4,
        )

        results = {
            "model_name": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f1_macro": f1_macro,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

        print(f"  准确率: {accuracy:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  Macro F1: {f1_macro:.4f}")

        return results, predictions

    def qualitative_analysis(self, test_data, all_predictions):
        """定性分析：展示困难样本"""
        print("\n🔬 定性分析：困难样本对比...")

        diff_samples = []

        for i, item in enumerate(test_data):
            true_label_id = self.label_text_to_id(
                item.get("output", item.get("sentiment"))
            )
            true_label = "positive" if true_label_id == 1 else "negative"

            preds = {name: preds[i] for name, preds in all_predictions.items()}

            # 如果预测不一致，或者（都一致但）和真实标签不一致
            if len(set(preds.values())) > 1 or list(preds.values())[0] != true_label_id:
                # 文本：优先用 review，没有就用 input
                review_text = item.get("review", item.get("input", ""))

                diff_samples.append({
                    "index": i,
                    "review": review_text,
                    "true_label": true_label,
                    "predictions": {
                        name: "positive" if p == 1 else "negative"
                        for name, p in preds.items()
                    },
                })

        # 保存分析结果
        output_file = self.results_dir / "qualitative_analysis.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(diff_samples[:20], f, ensure_ascii=False, indent=2)

        print(f"  发现 {len(diff_samples)} 个困难样本")
        print(f"  示例已保存到: {output_file}")

        # 打印几个示例
        print("\n示例分析（前5个）:")
        for i, sample in enumerate(diff_samples[:5]):
            print(f"\n样本 {i + 1}:")
            print(f"  评论: {sample['review'][:60]}...")
            print(f"  真实标签: {sample['true_label']}")
            for model_name, pred in sample["predictions"].items():
                correct = "✓" if pred == sample["true_label"] else "✗"
                print(f"  {model_name}: {pred} {correct}")

    def plot_comparison(self, all_results):
        """绘制对比图"""
        print("\n📊 绘制对比图...")

        models = list(all_results.keys())
        metrics = {
            "Accuracy": [all_results[m]["accuracy"] for m in models],
            "F1 Score": [all_results[m]["f1"] for m in models],
            "Macro F1": [all_results[m]["f1_macro"] for m in models],
        }

        df = pd.DataFrame(metrics, index=models)

        # 柱状图
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(models))
        width = 0.25

        bars1 = ax.bar(x - width, df["Accuracy"], width, label="Accuracy", alpha=0.8)
        bars2 = ax.bar(x, df["F1 Score"], width, label="F1 Score", alpha=0.8)
        bars3 = ax.bar(x + width, df["Macro F1"], width, label="Macro F1", alpha=0.8)

        ax.set_xlabel("Models", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim([0.7, 1.0])

        # 数值标签
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.tight_layout()
        plt.savefig(
            self.results_dir / "model_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"  ✅ 对比图已保存: {self.results_dir / 'model_comparison.png'}")

        plt.close()

        # 混淆矩阵
        fig, axes = plt.subplots(1, len(models), figsize=(15, 4))

        for idx, model_name in enumerate(models):
            cm = np.array(all_results[model_name]["confusion_matrix"])

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
                ax=axes[idx],
                cbar=False,
            )

            axes[idx].set_title(model_name, fontsize=11)
            axes[idx].set_xlabel("Predicted")
            axes[idx].set_ylabel("True")

        plt.tight_layout()
        plt.savefig(
            self.results_dir / "confusion_matrices.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(f"  ✅ 混淆矩阵已保存: {self.results_dir / 'confusion_matrices.png'}")

        plt.close()

    def generate_report(self, all_results):
        """生成评估报告"""
        print("\n📝 生成评估报告...")

        comparison_data = []

        for model_name, results in all_results.items():
            comparison_data.append({
                "Model": model_name,
                "Accuracy": f"{results['accuracy']:.4f}",
                "Precision": f"{results['precision']:.4f}",
                "Recall": f"{results['recall']:.4f}",
                "F1 Score": f"{results['f1']:.4f}",
                "Macro F1": f"{results['f1_macro']:.4f}",
            })

        df = pd.DataFrame(comparison_data)

        report_file = self.results_dir / "evaluation_report.md"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("# 模型评估报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 模型性能对比\n\n")

            # 手动生成 Markdown 表格，避免依赖 tabulate
            headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "Macro F1"]
            # 表头
            f.write("| " + " | ".join(headers) + " |\n")
            # 分隔行
            f.write("|" + "|".join([" --- " for _ in headers]) + "|\n")
            # 每一行数据
            for _, row in df.iterrows():
                f.write(
                    "| "
                    + " | ".join(
                        str(row[h]) for h in headers
                    )
                    + " |\n"
                )

            f.write("\n\n")

            f.write("## 详细分类报告\n\n")
            for model_name, results in all_results.items():
                f.write(f"### {model_name}\n\n")
                f.write("```\n")
                f.write(results["classification_report"])
                f.write("\n```\n\n")

            f.write("## 结论\n\n")

            # 找出最佳模型（按 weighted F1）
            best_model = max(all_results.items(), key=lambda x: x[1]["f1"])
            f.write(f"- **最佳模型**: {best_model[0]}\n")
            f.write(f"- **F1分数**: {best_model[1]['f1']:.4f}\n")
            f.write(f"- **准确率**: {best_model[1]['accuracy']:.4f}\n\n")

            if "BERT Baseline" in all_results and "BERT + SFT + DPO" in all_results:
                baseline_f1 = all_results["BERT Baseline"]["f1"]
                final_f1 = all_results["BERT + SFT + DPO"]["f1"]
                improvement = ((final_f1 - baseline_f1) / baseline_f1) * 100

                f.write(f"- **相对改进**: {improvement:.2f}%\n")
                f.write(f"  - Baseline F1: {baseline_f1:.4f}\n")
                f.write(f"  - Final F1: {final_f1:.4f}\n")

        print(f"  ✅ 报告已保存: {report_file}")

        json_file = self.results_dir / "all_results.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

        print(f"  ✅ JSON结果已保存: {json_file}")

    def run_evaluation(self):
        """运行完整评估"""
        print("\n" + "=" * 60)
        print("🚀 开始综合评估")
        print("=" * 60)

        test_data = self.load_test_data()

        all_results = {}
        all_predictions = {}

        for model_name, model_path in self.models.items():
            result = self.evaluate_model(model_name, model_path, test_data)
            if result is not None:
                results, predictions = result
                all_results[model_name] = results
                all_predictions[model_name] = predictions

        if not all_results:
            print("❌ 没有可用的模型进行评估")
            return

        self.qualitative_analysis(test_data, all_predictions)
        self.plot_comparison(all_results)
        self.generate_report(all_results)

        print("\n" + "=" * 60)
        print("✅ 综合评估完成！")
        print("=" * 60)
        print(f"\n📁 结果保存在: {self.results_dir}")


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run_evaluation()

    print("\n📌 下一步:")
    print("  运行: python scripts/6_demo_app.py")
