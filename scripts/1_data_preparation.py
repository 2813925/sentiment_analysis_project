#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据准备脚本 - 完整修复版
下载并处理ChnSentiCorp数据集 + 生成自构数据
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datasets import Dataset, DatasetDict

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DPO_DATA_DIR = DATA_DIR / "dpo_pairs"

# 创建必要的目录
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
DPO_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset():
    """下载ChnSentiCorp数据集"""
    print("📥 下载ChnSentiCorp数据集...")

    dataset_path = RAW_DATA_DIR / "ChnSentiCorp" / "ChnSentiCorp_htl_all.csv"

    if dataset_path.exists():
        print("✅ 数据集已存在，跳过下载")
        return dataset_path

    # 创建目录
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    # 提示用户手动下载
    print("\n⚠️  需要手动下载数据集！")
    print("请访问: https://github.com/SophonPlus/ChineseNlpCorpus")
    print("下载: ChnSentiCorp_htl_all.csv")
    print(f"保存到: {dataset_path}")

    # 尝试自动下载
    try:
        import requests
        url = "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv"

        print("🌐 尝试自动下载...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(dataset_path, 'wb') as f:
            f.write(response.content)
        print("✅ 数据集下载完成")

    except Exception as e:
        print(f"⚠️  自动下载失败: {str(e)}")
        print("请手动下载数据集")
        return None

    return dataset_path


def process_chnsenticorp(dataset_path):
    """处理ChnSentiCorp数据集"""
    print("\n📊 处理ChnSentiCorp数据集...")

    # 读取数据
    df = pd.read_csv(dataset_path)
    print(f"总样本数: {len(df)}")
    print(f"列名: {df.columns.tolist()}")

    # 数据清洗
    df = df.dropna()  # 删除缺失值
    df = df[df['review'].str.len() > 10]  # 过滤太短的评论
    print(f"清洗后样本数: {len(df)}")

    # 检查标签分布
    print(f"正面样本: {(df['label'] == 1).sum()}")
    print(f"负面样本: {(df['label'] == 0).sum()}")

    # 划分数据集
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
    )

    print(f"✅ 训练集: {len(train_df)} | 验证集: {len(val_df)} | 测试集: {len(test_df)}")

    # 重命名列以匹配transformers格式
    train_df = train_df.rename(columns={'review': 'text'})
    val_df = val_df.rename(columns={'review': 'text'})
    test_df = test_df.rename(columns={'review': 'text'})

    # 转换为datasets格式并保存
    print("\n💾 保存为 Datasets 格式...")
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']].reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']].reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']].reset_index(drop=True))

    # 保存为 datasets 格式（这是关键！）
    train_dataset.save_to_disk(str(PROCESSED_DATA_DIR / "train"))
    val_dataset.save_to_disk(str(PROCESSED_DATA_DIR / "validation"))
    test_dataset.save_to_disk(str(PROCESSED_DATA_DIR / "test"))

    print(f"✅ Datasets格式数据已保存到:")
    print(f"   - {PROCESSED_DATA_DIR / 'train'}")
    print(f"   - {PROCESSED_DATA_DIR / 'validation'}")
    print(f"   - {PROCESSED_DATA_DIR / 'test'}")

    return train_df, val_df, test_df


def generate_custom_data():
    """生成自构标注数据（包含复杂情感场景）"""
    print("\n🎨 生成自构标注数据（用于DPO训练）...")

    custom_samples = []

    # 1. 讽刺类（100条）
    sarcastic_templates = [
        ("这服务真是{adj}，让我{feel}得不行", 0),
        ("哇，{thing}真是{adj}啊，{result}", 0),
        ("嗯，{aspect}{adj}，我太{feel}了", 0),
        ("呵呵，{aspect}确实{adj}，{result}", 0),
        ("可真是{adj}的{thing}，我都{feel}了", 0),
    ]

    adj_negative = ["好", "棒", "优秀", "完美", "贴心", "周到", "细致"]
    feel_negative = ["失望", "生气", "无语", "郁闷", "愤怒", "寒心"]
    things = ["体验", "质量", "态度", "效率", "服务"]
    aspects = ["服务", "产品", "环境", "速度", "态度"]
    results = ["真让人失望", "完全不行", "气死我了", "太糟糕了", "无法接受"]

    for _ in range(100):
        template, label = sarcastic_templates[np.random.randint(0, len(sarcastic_templates))]
        text = template.format(
            adj=np.random.choice(adj_negative),
            feel=np.random.choice(feel_negative),
            thing=np.random.choice(things),
            aspect=np.random.choice(aspects),
            result=np.random.choice(results)
        )
        custom_samples.append({"text": text, "label": label})

    # 2. 隐喻类（100条）
    metaphor_samples = [
                           ("住进这家酒店就像回到了八十年代", 0),
                           ("服务员的态度能冻死人", 0),
                           ("这个房间简直是桑拿房", 0),
                           ("床硬得像睡在地板上", 0),
                           ("隔音效果等于零", 0),
                           ("卫生间小得转不开身", 0),
                           ("空调声音大得像拖拉机", 0),
                           ("早餐难吃得像猪食", 0),
                           ("网速慢得像蜗牛", 0),
                           ("价格贵得离谱", 0),
                           ("早餐丰富得像满汉全席", 1),
                           ("房间干净得像新装修的", 1),
                           ("服务员热情得像春风", 1),
                           ("睡眠质量好得像在家里", 1),
                           ("性价比高得令人惊喜", 1),
                           ("地理位置好得没话说", 1),
                           ("装修豪华得像五星级", 1),
                           ("床舒服得像云朵", 1),
                           ("服务周到得无可挑剔", 1),
                           ("环境优雅得像度假村", 1),
                       ] * 5

    for text, label in metaphor_samples:
        custom_samples.append({"text": text, "label": label})

    # 3. 双重否定（100条）
    double_negative = [
                          ("不得不说，这家酒店不差", 1),
                          ("说实话，没有什么不满意的", 1),
                          ("不能说不好，但也不是特别好", 0),
                          ("并非不能接受，但确实不太满意", 0),
                          ("没什么不好的，就是价格不便宜", 0),
                          ("不是不推荐，只是性价比不高", 0),
                          ("不能说服务不好，但也谈不上热情", 0),
                          ("不是完全不能住，但下次不会再来", 0),
                          ("没有不干净，但也算不上整洁", 0),
                          ("不得不承认，确实不错", 1),
                          ("不能否认，这里很棒", 1),
                          ("说不上不喜欢，还挺满意的", 1),
                          ("没有不推荐的理由", 1),
                          ("不是没有缺点，但瑕不掩瑜", 1),
                      ] * 7 + [
                          ("并非完美无缺，但整体不错", 1),
                          ("不是说没有问题，但可以接受", 1),
                      ] * 3

    for text, label in double_negative[:100]:
        custom_samples.append({"text": text, "label": label})

    # 4. 对比转折（100条）
    contrast_templates = [
        ("虽然{positive}，但是{negative}", 0),
        ("{negative}，不过{positive}", 1),
        ("本来{negative}，结果{positive}", 1),
        ("{positive}，可惜{negative}", 0),
        ("除了{negative}，其他都{positive}", 1),
        ("整体{positive}，就是{negative}", 0),
    ]

    positives = ["服务很好", "环境不错", "位置便利", "房间干净", "设施齐全", "性价比高"]
    negatives = ["价格太贵", "隔音很差", "设施陈旧", "早餐难吃", "房间太小", "网络很慢"]

    for _ in range(100):
        template, label = contrast_templates[np.random.randint(0, len(contrast_templates))]
        text = template.format(
            positive=np.random.choice(positives),
            negative=np.random.choice(negatives)
        )
        custom_samples.append({"text": text, "label": label})

    # 5. 细粒度情感（100条）
    fine_grained = [
                       ("总体还可以，就是有些小瑕疵", 1),
                       ("基本满意，性价比尚可", 1),
                       ("一般般，没什么特别的", 0),
                       ("勉强能住，凑合一晚", 0),
                       ("还行吧，下次可能不会再来", 0),
                       ("中规中矩，符合预期", 1),
                       ("尚可接受，但谈不上惊喜", 0),
                       ("超出预期，相当不错", 1),
                       ("差强人意，勉强及格", 0),
                       ("物有所值，推荐入住", 1),
                       ("性价比一般，不太推荐", 0),
                       ("整体满意，会再次光顾", 1),
                       ("体验平平，没有亮点", 0),
                       ("相当满意，值得推荐", 1),
                       ("略显失望，期望过高", 0),
                   ] * 7 + [
                       ("整体不错，有待改进", 1),
                   ] * 5

    for text, label in fine_grained[:100]:
        custom_samples.append({"text": text, "label": label})

    # 保存
    custom_data_path = RAW_DATA_DIR / "custom_data.json"
    with open(custom_data_path, 'w', encoding='utf-8') as f:
        json.dump(custom_samples, f, ensure_ascii=False, indent=2)

    print(f"✅ 生成了 {len(custom_samples)} 条自构数据: {custom_data_path}")

    return custom_samples


def create_sft_data(train_df, val_df, test_df):
    """创建SFT指令格式数据"""
    instruction = "请判断以下酒店评论的情感倾向，回答'正面'或'负面'。"

    def convert_to_sft_format(df, output_file):
        print(f"🎯 创建SFT指令格式数据: {output_file}")
        sft_data = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="转换格式"):
            text = row['text']
            label = "正面" if row['label'] == 1 else "负面"

            sample = {
                "instruction": instruction,
                "input": text,
                "output": label
            }
            sft_data.append(sample)

        # 保存
        output_path = PROCESSED_DATA_DIR / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sft_data, f, ensure_ascii=False, indent=2)

        print(f"✅ 生成 {len(sft_data)} 条SFT数据")
        return sft_data

    # 转换三个数据集
    train_sft = convert_to_sft_format(train_df, "sft_train.json")
    val_sft = convert_to_sft_format(val_df, "sft_dev.json")
    test_sft = convert_to_sft_format(test_df, "sft_test.json")

    return train_sft, val_sft, test_sft


def create_dpo_pairs(custom_samples):
    """创建DPO偏好对数据"""
    print("\n🔄 创建DPO偏好对...")

    dpo_pairs = []

    # 对于每个样本，创建chosen和rejected响应
    for i in range(min(200, len(custom_samples))):
        sample = custom_samples[i]

        correct_label = "正面" if sample['label'] == 1 else "负面"
        wrong_label = "负面" if sample['label'] == 1 else "正面"

        dpo_pair = {
            "prompt": f"请判断以下酒店评论的情感倾向：{sample['text']}",
            "chosen": correct_label,
            "rejected": wrong_label
        }
        dpo_pairs.append(dpo_pair)

    # 保存
    dpo_train_path = DPO_DATA_DIR / "dpo_train.json"
    with open(dpo_train_path, 'w', encoding='utf-8') as f:
        json.dump(dpo_pairs, f, ensure_ascii=False, indent=2)

    print(f"✅ 生成 {len(dpo_pairs)} 对DPO训练数据: {dpo_train_path}")

    return dpo_pairs


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 开始数据准备流程")
    print("=" * 60)

    # 1. 下载数据集
    dataset_path = download_dataset()
    if dataset_path is None or not dataset_path.exists():
        print("\n❌ 数据集不存在，请手动下载！")
        print(f"下载链接: https://github.com/SophonPlus/ChineseNlpCorpus")
        print(f"保存位置: {RAW_DATA_DIR / 'ChnSentiCorp' / 'ChnSentiCorp_htl_all.csv'}")
        return

    # 2. 处理ChnSentiCorp数据
    train_df, val_df, test_df = process_chnsenticorp(dataset_path)

    # 3. 生成自构数据
    custom_samples = generate_custom_data()

    # 4. 创建SFT数据
    train_sft, val_sft, test_sft = create_sft_data(train_df, val_df, test_df)

    # 5. 创建DPO数据
    dpo_pairs = create_dpo_pairs(custom_samples)

    print("\n" + "=" * 60)
    print("✅ 数据准备完成！")
    print("=" * 60)
    print("\n📁 数据保存位置:")
    print(f"  - Datasets格式: {PROCESSED_DATA_DIR}")
    print(f"  - SFT数据: {PROCESSED_DATA_DIR}/sft_*.json")
    print(f"  - DPO数据: {DPO_DATA_DIR}/dpo_train.json")

    # 验证生成的文件
    print("\n🔍 验证生成的文件:")
    train_exists = (PROCESSED_DATA_DIR / 'train').exists()
    val_exists = (PROCESSED_DATA_DIR / 'validation').exists()
    test_exists = (PROCESSED_DATA_DIR / 'test').exists()

    print(f"  ✅ train dataset: {train_exists}")
    print(f"  ✅ validation dataset: {val_exists}")
    print(f"  ✅ test dataset: {test_exists}")

    if not (train_exists and val_exists and test_exists):
        print("\n⚠️  警告：部分数据集未成功生成！")
    else:
        print("\n🎉 所有数据集生成成功，可以开始训练了！")


if __name__ == "__main__":
    main()