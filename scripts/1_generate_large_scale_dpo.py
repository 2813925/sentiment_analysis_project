#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPO数据大规模生成 - 1500+条平衡数据
目标: 解决数据量不足和分布不平衡问题
"""

import json
import random
from pathlib import Path
from typing import List, Dict

class LargeScaleDPOGenerator:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent if Path(__file__).parent.name == 'scripts' else Path.cwd()
        self.data_dir = self.project_root / "data"
        
        # 扩展词库
        self.aspects = [
            "服务", "环境", "位置", "设施", "卫生", "性价比", "态度", "效率",
            "质量", "体验", "氛围", "装修", "交通", "停车", "网络", "早餐",
            "床品", "隔音", "采光", "温度", "热水", "空调", "电视", "wifi"
        ]
        
        self.positive_adj = [
            "很好", "不错", "优秀", "满意", "舒适", "给力", "赞", "棒",
            "完美", "优质", "贴心", "周到", "热情", "专业", "及时", "方便",
            "干净", "整洁", "宽敞", "明亮", "温馨", "舒服", "到位", "靠谱"
        ]
        
        self.negative_adj = [
            "很差", "不好", "糟糕", "失望", "难受", "不满", "烂", "坑",
            "恶劣", "低劣", "冷漠", "敷衍", "拖沓", "业余", "迟缓", "麻烦",
            "脏", "乱", "狭小", "昏暗", "压抑", "难受", "不到位", "不靠谱"
        ]
        
    def generate_simple_positive(self, n: int = 500) -> List[Dict]:
        """生成500条简单正面样本"""
        samples = []
        templates = [
            # 单方面正面 (200条)
            "{}{}，很满意",
            "{}非常{}",
            "对{}很满意，{}",
            "{}做得很{}",
            "{}相当{}",
            "{}让人满意",
            "{}值得称赞",
            "{}超出期待",
            
            # 推荐类 (150条)
            "强烈推荐，{}很{}",
            "值得推荐，{}非常{}",
            "推荐入住，{}",
            "下次还会来，{}很好",
            "会推荐给朋友",
            "性价比很高",
            "物有所值",
            "超值体验",
            
            # 多方面正面 (150条)
            "{}和{}都很好",
            "{}{}，{}也{}",
            "{}、{}都不错",
            "{}满意，{}也满意",
        ]
        
        for _ in range(n):
            if len(samples) < 200:  # 单方面正面
                template = random.choice(templates[:8])
                aspect = random.choice(self.aspects)
                adj = random.choice(self.positive_adj)
                
                if template.count('{}') == 2:
                    chosen = template.format(aspect, adj)
                else:
                    chosen = template.format(aspect)
            
            elif len(samples) < 350:  # 推荐类
                template = random.choice(templates[8:16])
                if template.count('{}') == 2:
                    aspect = random.choice(self.aspects)
                    adj = random.choice(self.positive_adj)
                    chosen = template.format(aspect, adj)
                elif template.count('{}') == 1:
                    aspect = random.choice(self.aspects)
                    chosen = template.format(aspect)
                else:
                    chosen = template
            
            else:  # 多方面正面
                template = random.choice(templates[16:])
                aspects = random.sample(self.aspects, 2)
                adjs = random.sample(self.positive_adj, 2)
                
                if template.count('{}') == 4:
                    chosen = template.format(aspects[0], adjs[0], aspects[1], adjs[1])
                elif template.count('{}') == 3:
                    chosen = template.format(aspects[0], aspects[1], adjs[0])
                else:
                    chosen = template.format(aspects[0], aspects[1])
            
            # 生成rejected（负面版本）
            rejected = self._flip_sentiment(chosen, positive_to_negative=True)
            
            samples.append({
                'prompt': f'分析这段评论的情感倾向："{chosen}"',
                'chosen': '正面',
                'rejected': '负面'
            })
        
        return samples
    
    def generate_simple_negative(self, n: int = 500) -> List[Dict]:
        """生成500条简单负面样本"""
        samples = []
        templates = [
            # 单方面负面 (200条)
            "{}{}，很失望",
            "{}非常{}",
            "对{}很不满，{}",
            "{}做得很{}",
            "{}相当{}",
            "{}让人失望",
            "{}需要改进",
            "{}不符合期待",
            
            # 不推荐类 (150条)
            "不推荐，{}很{}",
            "不建议入住，{}非常{}",
            "不会再来了，{}",
            "下次不会选择这里，{}很差",
            "不会推荐给朋友",
            "性价比太低",
            "不值这个价",
            "糟糕的体验",
            
            # 多方面负面 (150条)
            "{}和{}都很差",
            "{}{}，{}也{}",
            "{}、{}都不行",
            "{}不满意，{}也不满意",
        ]
        
        for _ in range(n):
            if len(samples) < 200:
                template = random.choice(templates[:8])
                aspect = random.choice(self.aspects)
                adj = random.choice(self.negative_adj)
                
                if template.count('{}') == 2:
                    chosen = template.format(aspect, adj)
                else:
                    chosen = template.format(aspect)
            
            elif len(samples) < 350:
                template = random.choice(templates[8:16])
                if template.count('{}') == 2:
                    aspect = random.choice(self.aspects)
                    adj = random.choice(self.negative_adj)
                    chosen = template.format(aspect, adj)
                elif template.count('{}') == 1:
                    aspect = random.choice(self.aspects)
                    chosen = template.format(aspect)
                else:
                    chosen = template
            
            else:
                template = random.choice(templates[16:])
                aspects = random.sample(self.aspects, 2)
                adjs = random.sample(self.negative_adj, 2)
                
                if template.count('{}') == 4:
                    chosen = template.format(aspects[0], adjs[0], aspects[1], adjs[1])
                elif template.count('{}') == 3:
                    chosen = template.format(aspects[0], aspects[1], adjs[0])
                else:
                    chosen = template.format(aspects[0], aspects[1])
            
            rejected = self._flip_sentiment(chosen, positive_to_negative=False)
            
            samples.append({
                'prompt': f'分析这段评论的情感倾向："{chosen}"',
                'chosen': '负面',
                'rejected': '正面'
            })
        
        return samples
    
    def generate_complex_samples(self, n: int = 300) -> List[Dict]:
        """生成300条困难样本（讽刺、转折、双重否定等）"""
        samples = []
        
        # 讽刺样本 (100条)
        sarcasm_templates = [
            "{}是{}，如果你对{}没有要求的话",
            "{}还{}，就是{}了点",
            "除了{}不行，其他都挺好",
            "{}到是{}，可惜{}",
            "要不是{}，{}还是挺不错的",
        ]
        
        for _ in range(100):
            template = random.choice(sarcasm_templates)
            aspect = random.choice(self.aspects)
            adj_pos = random.choice(self.positive_adj)
            adj_neg = random.choice(self.negative_adj)
            
            # 50%正面讽刺，50%负面讽刺
            if random.random() < 0.5:
                chosen_text = template.format(aspect, adj_pos, aspect)
                chosen_label = '负面'
                rejected_label = '正面'
            else:
                chosen_text = template.format(aspect, adj_neg, aspect)
                chosen_label = '正面'
                rejected_label = '负面'
            
            samples.append({
                'prompt': f'分析这段评论的情感倾向："{chosen_text}"',
                'chosen': chosen_label,
                'rejected': rejected_label
            })
        
        # 转折样本 (100条)
        contrast_templates = [
            "{}虽然{}，但是{}",
            "{}是{}，不过{}有点{}",
            "整体{}，就是{}需要改进",
            "总的来说{}，但{}",
        ]
        
        for _ in range(100):
            template = random.choice(contrast_templates)
            aspects = random.sample(self.aspects, 2)
            adj_pos = random.choice(self.positive_adj)
            adj_neg = random.choice(self.negative_adj)
            
            if random.random() < 0.5:
                if template.count('{}') == 4:
                    chosen_text = template.format(aspects[0], adj_pos, aspects[1], adj_neg)
                else:
                    chosen_text = template.format(aspects[0], adj_pos, aspects[1])
                chosen_label = random.choice(['正面', '负面'])
            else:
                if template.count('{}') == 4:
                    chosen_text = template.format(aspects[0], adj_neg, aspects[1], adj_pos)
                else:
                    chosen_text = template.format(aspects[0], adj_neg, aspects[1])
                chosen_label = random.choice(['正面', '负面'])
            
            rejected_label = '负面' if chosen_label == '正面' else '正面'
            
            samples.append({
                'prompt': f'分析这段评论的情感倾向："{chosen_text}"',
                'chosen': chosen_label,
                'rejected': rejected_label
            })
        
        # 双重否定 (100条)
        double_neg_templates = [
            "不是说{}不{}",
            "{}不能说不{}",
            "{}也不是不能接受",
            "并非{}不{}",
        ]
        
        for _ in range(100):
            template = random.choice(double_neg_templates)
            aspect = random.choice(self.aspects)
            adj = random.choice(self.positive_adj)
            
            chosen_text = template.format(aspect, adj) if template.count('{}') == 2 else template.format(aspect)
            chosen_label = random.choice(['正面', '负面'])
            rejected_label = '负面' if chosen_label == '正面' else '正面'
            
            samples.append({
                'prompt': f'分析这段评论的情感倾向："{chosen_text}"',
                'chosen': chosen_label,
                'rejected': rejected_label
            })
        
        return samples
    
    def _flip_sentiment(self, text: str, positive_to_negative: bool) -> str:
        """翻转情感（生成rejected样本）"""
        # 简单翻转策略
        replacements = {
            # 正面 -> 负面
            '很好': '很差', '不错': '不好', '满意': '失望', '舒适': '难受',
            '给力': '不行', '赞': '差', '棒': '糟', '优秀': '糟糕',
            # 负面 -> 正面
            '很差': '很好', '不好': '不错', '失望': '满意', '难受': '舒适',
            '不行': '给力', '差': '赞', '糟': '棒', '糟糕': '优秀',
        }
        
        result = text
        if positive_to_negative:
            for pos, neg in list(replacements.items())[:8]:
                result = result.replace(pos, neg)
        else:
            for neg, pos in list(replacements.items())[8:]:
                result = result.replace(neg, pos)
        
        return result
    
    def load_original_data(self) -> List[Dict]:
        """加载原始困难样本"""
        original_file = self.data_dir / "dpo_pairs" / "dpo_train.json"
        if original_file.exists():
            with open(original_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def generate_large_scale_data(self):
        """生成大规模平衡数据"""
        print("="*60)
        print("🚀 DPO大规模数据生成 - 1500+条")
        print("="*60)
        
        # 1. 加载原始困难样本
        print("\n📊 Part 1: 加载原始困难样本...")
        original_samples = self.load_original_data()
        print(f"   ✅ 原始困难样本: {len(original_samples)}条")
        
        # 2. 生成简单正面样本
        print("\n📊 Part 2: 生成简单正面样本...")
        positive_samples = self.generate_simple_positive(500)
        print(f"   ✅ 简单正面样本: {len(positive_samples)}条")
        
        # 3. 生成简单负面样本
        print("\n📊 Part 3: 生成简单负面样本...")
        negative_samples = self.generate_simple_negative(500)
        print(f"   ✅ 简单负面样本: {len(negative_samples)}条")
        
        # 4. 生成复杂样本
        print("\n📊 Part 4: 生成复杂样本...")
        complex_samples = self.generate_complex_samples(300)
        print(f"   ✅ 复杂样本: {len(complex_samples)}条")
        
        # 5. 合并所有数据
        all_samples = original_samples + positive_samples + negative_samples + complex_samples
        
        # 6. 去重
        unique_samples = []
        seen_prompts = set()
        for sample in all_samples:
            prompt_key = sample['prompt']
            if prompt_key not in seen_prompts:
                seen_prompts.add(prompt_key)
                unique_samples.append(sample)
        
        # 7. 打乱顺序
        random.seed(42)
        random.shuffle(unique_samples)
        
        # 8. 统计
        print("\n"+"="*60)
        print("📊 最终数据集统计")
        print("="*60)
        print(f"  原始困难样本: {len(original_samples)}条")
        print(f"  简单正面样本: {len(positive_samples)}条")
        print(f"  简单负面样本: {len(negative_samples)}条")
        print(f"  新增复杂样本: {len(complex_samples)}条")
        print(f"  去重前总数: {len(all_samples)}条")
        print(f"  去重后总数: {len(unique_samples)}条")
        
        # 统计标签分布
        chosen_pos = len([s for s in unique_samples if s['chosen'] == '正面'])
        chosen_neg = len([s for s in unique_samples if s['chosen'] == '负面'])
        
        print(f"\n✅ 标签分布:")
        print(f"  Chosen 正面: {chosen_pos} ({chosen_pos/len(unique_samples)*100:.1f}%)")
        print(f"  Chosen 负面: {chosen_neg} ({chosen_neg/len(unique_samples)*100:.1f}%)")
        
        # 9. 保存
        output_file = self.data_dir / "dpo_pairs" / "dpo_train_large_scale.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(unique_samples, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 大规模数据已保存到: {output_file}")
        
        # 10. 生成统计报告
        stats = {
            'total': len(unique_samples),
            'original_hard': len(original_samples),
            'simple_positive': len(positive_samples),
            'simple_negative': len(negative_samples),
            'complex_new': len(complex_samples),
            'chosen_positive': chosen_pos,
            'chosen_negative': chosen_neg,
            'balance_ratio': chosen_pos / chosen_neg if chosen_neg > 0 else 0
        }
        
        stats_file = self.data_dir / "dpo_pairs" / "large_scale_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"📊 统计信息已保存到: {stats_file}")
        
        print("\n"+"="*60)
        print("✅ 大规模数据生成完成！")
        print("="*60)
        
        print("\n📌 下一步:")
        print("  1. 数据已自动保存为: dpo_train_large_scale.json")
        print("  2. 修改训练脚本使用新数据:")
        print("     sed -i 's/dpo_train_balanced.json/dpo_train_large_scale.json/g' scripts/4_dpo_training.py")
        print("  3. 调整训练配置（建议）:")
        print("     - learning_rate: 5e-5 → 3e-5 (数据多了，可以降低)")
        print("     - num_train_epochs: 10 → 5 (数据多了，不需要太多epoch)")
        print("     - batch_size: 16 → 32 (数据多了，可以增大)")
        print("  4. 运行训练:")
        print("     python scripts/4_dpo_training.py --local")
        
        print(f"\n📊 预期效果: Test F1 > 90% (有了{len(unique_samples)}条平衡数据)")
        
        return unique_samples

if __name__ == "__main__":
    generator = LargeScaleDPOGenerator()
    generator.generate_large_scale_data()
