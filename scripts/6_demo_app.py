"""
Gradio Demo应用 - Step 6
交互式情感分析演示
"""

import os
import torch
import gradio as gr
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification

# 只使用本地模型，避免去 HuggingFace 联网
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class SentimentAnalysisDemo:
    def __init__(self, base_dir: str = "./"):
        # 使用绝对路径，防止工作目录变化导致路径错误
        self.base_dir = Path(base_dir).resolve()
        self.models_dir = self.base_dir / "models"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  使用设备: {self.device}")

        # 加载所有可用模型
        self.models = {}
        self.tokenizers = {}

        self.load_models()

        self.max_length = 256

    def load_models(self):
        """加载所有训练好的模型"""
        print("\n📦 加载模型...")

        # Baseline 使用 final 目录，其余使用各自目录
        model_configs = {
            "BERT Baseline": self.models_dir / "bert_baseline" / "final",
            "BERT + SFT": self.models_dir / "bert_sft",
            "BERT + SFT + DPO": self.models_dir / "bert_dpo",
        }

        for name, path in model_configs.items():
            if path.exists():
                try:
                    tokenizer = BertTokenizer.from_pretrained(
                        str(path),
                        local_files_only=True,
                    )
                    model = BertForSequenceClassification.from_pretrained(
                        str(path),
                        local_files_only=True,
                    )
                    model.to(self.device)
                    model.eval()

                    self.models[name] = model
                    self.tokenizers[name] = tokenizer

                    print(f"  ✅ {name}")
                except Exception as e:
                    print(f"  ❌ {name}: {e}")
            else:
                print(f"  ⚠️  {name}: 模型不存在 -> {path}")

        if not self.models:
            print("\n⚠️  警告: 没有找到训练好的模型！")
            print("请先运行训练脚本:")
            print("  python scripts/2_baseline_training.py")
            print("  python scripts/3_sft_training.py")
            print("  python scripts/4_dpo_training.py")

    def predict(self, text, model_name):
        """预测单个文本"""
        if model_name not in self.models:
            return "❌ 模型未加载", None

        if not text.strip():
            return "请输入评论文本", None

        # 获取模型和tokenizer
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]

        # 构造输入（SFT/DPO 用指令格式，Baseline 直接用原始文本）
        if model_name != "BERT Baseline":
            instruction = "任务：判断以下评论的情感倾向（正面/负面），并简要说明理由。"
            input_text = f"{instruction}\n评论：{text}\n答案："
        else:
            input_text = text

        # Tokenize
        inputs = tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        # 预测
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = torch.argmax(logits, dim=1).item()

        # 结果
        sentiment = "😊 正面" if pred == 1 else "😞 负面"
        confidence = probs[pred] * 100

        # 详细信息
        result_text = f"""
### 分析结果

**情感倾向**: {sentiment}  
**置信度**: {confidence:.2f}%

**详细概率**:
- 负面: {probs[0] * 100:.2f}%
- 正面: {probs[1] * 100:.2f}%
"""

        # 返回概率分布（用于图表）
        prob_dict = {
            "负面": float(probs[0]),
            "正面": float(probs[1]),
        }

        return result_text, prob_dict

    def predict_all_models(self, text):
        """使用所有模型预测"""
        if not text.strip():
            return "请输入评论文本"

        results = "## 所有模型对比结果\n\n"

        for model_name in self.models.keys():
            result, _ = self.predict(text, model_name)
            results += f"### {model_name}\n{result}\n---\n\n"

        return results

    def create_interface(self):
        """创建Gradio界面"""

        # 示例文本
        examples = [
            ["这家店的服务真是太贴心了，每个细节都考虑到了！"],
            ["产品质量超出预期，物超所值，强烈推荐！"],
            ["这什么破玩意儿，用了两天就坏了，太失望了。"],
            ["客服态度极差，问题迟迟得不到解决。"],
            ["真是'物美价廉'啊，一分钱一分货都不如。"],
            ["还行吧，凑合能用。"],
        ]

        # 主题CSS
        custom_css = """
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .output-markdown {
            font-size: 16px;
        }
        """

        with gr.Blocks(
            title="中文评论情感分析",
            css=custom_css,
            theme=gr.themes.Soft()
        ) as demo:

            gr.Markdown(
                """
            # 🎯 中文评论情感分析系统
            
            基于 BERT 的情感分析模型，支持 **SFT** 和 **DPO** 微调
            
            ---
            """
            )

            # 单模型预测
            with gr.Tab("单模型预测"):
                with gr.Row():
                    with gr.Column(scale=2):
                        input_text = gr.Textbox(
                            label="输入评论",
                            placeholder="请输入中文评论文本...",
                            lines=3,
                        )

                        model_choice = gr.Dropdown(
                            choices=list(self.models.keys()),
                            value=list(self.models.keys())[0]
                            if self.models
                            else None,
                            label="选择模型",
                        )

                        predict_btn = gr.Button("🔍 分析情感", variant="primary")

                    with gr.Column(scale=3):
                        output_text = gr.Markdown(label="分析结果")
                        # 用 Label 展示概率
                        output_plot = gr.Label(
                            label="概率分布", num_top_classes=2
                        )

                gr.Examples(
                    examples=examples,
                    inputs=input_text,
                    label="示例评论",
                )

                predict_btn.click(
                    fn=self.predict,
                    inputs=[input_text, model_choice],
                    outputs=[output_text, output_plot],
                )

            # 模型对比
            with gr.Tab("模型对比"):
                with gr.Row():
                    with gr.Column(scale=1):
                        compare_input = gr.Textbox(
                            label="输入评论",
                            placeholder="请输入中文评论文本...",
                            lines=4,
                        )
                        compare_btn = gr.Button(
                            "🔄 对比所有模型", variant="primary"
                        )

                    with gr.Column(scale=2):
                        compare_output = gr.Markdown(label="对比结果")

                gr.Examples(
                    examples=examples,
                    inputs=compare_input,
                    label="示例评论",
                )

                compare_btn.click(
                    fn=self.predict_all_models,
                    inputs=compare_input,
                    outputs=compare_output,
                )

            # 项目说明
            with gr.Tab("项目说明"):
                gr.Markdown(
                    """
                ## 📖 项目介绍
                
                本项目实现了基于 BERT 的中文评论情感分析，采用以下技术路线：
                
                ### 🔧 技术栈
                
                1. **Baseline**: BERT-base-chinese
                2. **SFT (Supervised Fine-Tuning)**: 指令式微调
                3. **DPO (Direct Preference Optimization)**: 偏好优化（简化版）
                
                ### 📊 数据集
                
                - **ChnSentiCorp**: 中文情感分析语料
                - **偏好对数据**: 约 200 条人工标注的偏好对（prompt + chosen + rejected）
                
                ### 🎯 在测试集上的实际性能（本项目实测结果）
                
                | 模型 | Accuracy | F1 Score | Macro F1 |
                |------|----------|----------|----------|
                | BERT Baseline | 0.8943 | 0.8966 | 0.8832 |
                | BERT + SFT | 0.9188 | 0.9188 | 0.9057 |
                | BERT + SFT + DPO | 0.9137 | 0.9150 | 0.9032 |
                
                ### 🌟 项目亮点
                
                - ✅ 完整的训练 pipeline（Baseline → SFT → DPO）
                - ✅ 利用偏好对数据进行对齐微调
                - ✅ 详细的模型对比和可视化（混淆矩阵、指标对比）
                - ✅ 交互式 Gradio Demo 应用
                
                ---
                
                本 Demo 用于课程/项目展示，方便直观体验不同训练阶段模型的效果差异。
                """
                )

        return demo

    def launch(self, share: bool = False):
        """启动应用"""
        if not self.models:
            print("\n❌ 没有可用的模型！")
            print("请先训练模型后再启动Demo。")
            return

        demo = self.create_interface()

        print("\n" + "=" * 60)
        print("🚀 启动 Gradio Demo")
        print("=" * 60)
        print(f"已加载 {len(self.models)} 个模型")
        print("\n访问地址将在浏览器中显示（注意使用 127.0.0.1 或 服务器IP 访问，不要用 0.0.0.0）")

        demo.launch(
            share=share,
            server_name="0.0.0.0",  # 监听所有网卡
            server_port=7860,
            show_error=True,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--share", action="store_true", help="创建公开分享链接"
    )
    args = parser.parse_args()

    demo_app = SentimentAnalysisDemo()
    demo_app.launch(share=args.share)
