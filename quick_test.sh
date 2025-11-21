#!/bin/bash

# 快速测试脚本 - 使用小规模参数快速验证流程
# 适合测试环境或快速演示

echo "==============================================="
echo "⚡ 快速测试模式"
echo "==============================================="
echo "这个脚本将使用较少的训练轮数快速完成整个流程"
echo "适合用于："
echo "  - 测试环境是否正确配置"
echo "  - 快速验证代码是否能正常运行"
echo "  - 演示完整流程"
echo ""
echo "注意: 快速模式的模型性能会比完整训练差"
echo "==============================================="
echo ""

read -p "是否继续? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "🚀 开始快速测试..."
echo ""

# 数据准备
echo "📊 [1/5] 数据准备..."
python scripts/data_preparation.py
if [ $? -ne 0 ]; then
    echo "❌ 失败"
    exit 1
fi
echo "✅ 完成"

# 修改训练参数为快速模式
echo ""
echo "📚 [2/5] Baseline训练 (快速模式: 1 epoch)..."
python -c "
import sys
sys.path.append('.')
from scripts.train_baseline import BERTTrainer

trainer = BERTTrainer(
    model_name='bert-base-chinese',
    num_labels=3,
    output_dir='./models/bert_baseline'
)
trainer.load_data()
trainer.train(epochs=1, batch_size=32, learning_rate=2e-5)
trainer.load_model('./models/bert_baseline/best_model')
trainer.test()
"
if [ $? -ne 0 ]; then
    echo "❌ 失败"
    exit 1
fi
echo "✅ 完成"

echo ""
echo "🎯 [3/5] SFT训练 (快速模式: 2 epochs)..."
python -c "
import sys
sys.path.append('.')
from scripts.train_sft import SFTTrainer

trainer = SFTTrainer(
    base_model_path='./models/bert_baseline/best_model',
    num_labels=3,
    output_dir='./models/bert_sft'
)
trainer.load_data()
trainer.train(epochs=2, batch_size=16, learning_rate=2e-5)
trainer.load_model('./models/bert_sft/best_model')
trainer.test()
"
if [ $? -ne 0 ]; then
    echo "❌ 失败"
    exit 1
fi
echo "✅ 完成"

echo ""
echo "⚖️  [4/5] DPO训练 (快速模式: 1 epoch)..."
python -c "
import sys
sys.path.append('.')
from scripts.train_dpo import DPOTrainer

trainer = DPOTrainer(
    sft_model_path='./models/bert_sft/best_model',
    num_labels=3,
    output_dir='./models/bert_dpo',
    beta=0.1
)
trainer.load_data()
trainer.train(epochs=1, batch_size=8, learning_rate=5e-6)
trainer.load_model('./models/bert_dpo/best_model')
trainer.test()
"
if [ $? -ne 0 ]; then
    echo "❌ 失败"
    exit 1
fi
echo "✅ 完成"

echo ""
echo "📊 [5/5] 模型评估..."
python scripts/evaluate.py
if [ $? -ne 0 ]; then
    echo "❌ 失败"
    exit 1
fi
echo "✅ 完成"

echo ""
echo "==============================================="
echo "✅ 快速测试完成！"
echo "==============================================="
echo ""
echo "📊 查看结果:"
echo "   cat evaluation_results/detailed_report.txt"
echo ""
echo "🚀 启动Demo:"
echo "   python demo/gradio_app.py"
echo ""
