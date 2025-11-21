"""
一键运行完整项目流程
适合演示和快速测试
"""

import subprocess
import sys
from pathlib import Path
import time

class ProjectRunner:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.scripts_dir = self.base_dir / "scripts"
        
    def run_script(self, script_name, description):
        """运行单个脚本"""
        print("\n" + "="*70)
        print(f"🚀 {description}")
        print("="*70)
        
        script_path = self.scripts_dir / script_name
        
        if not script_path.exists():
            print(f"❌ 脚本不存在: {script_path}")
            return False
        
        try:
            # 运行脚本
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.base_dir),
                check=True,
                capture_output=False,
                text=True
            )
            
            print(f"\n✅ {description} - 完成!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ {description} - 失败!")
            print(f"错误: {e}")
            return False
        except KeyboardInterrupt:
            print(f"\n⚠️  用户中断")
            return False
    
    def run_all(self, skip_training=False):
        """运行完整流程"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     中文评论情感分析项目 - 自动运行脚本                      ║
║                                                              ║
║     BERT + SFT + DPO 完整Pipeline                           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
        
        start_time = time.time()
        
        # 步骤1: 数据准备
        if not self.run_script("1_data_preparation.py", "Step 1: 数据准备"):
            print("\n⚠️  数据准备失败，请检查数据集是否下载")
            return
        
        if skip_training:
            print("\n⚠️  跳过训练步骤（--skip-training）")
        else:
            # 步骤2: Baseline训练
            if not self.run_script("2_baseline_training.py", "Step 2: Baseline模型训练"):
                print("\n⚠️  Baseline训练失败")
                response = input("是否继续? (y/n): ")
                if response.lower() != 'y':
                    return
            
            # 步骤3: SFT训练
            if not self.run_script("3_sft_training.py", "Step 3: SFT微调"):
                print("\n⚠️  SFT训练失败")
                response = input("是否继续? (y/n): ")
                if response.lower() != 'y':
                    return
            
            # 步骤4: DPO训练
            if not self.run_script("4_dpo_training.py", "Step 4: DPO微调"):
                print("\n⚠️  DPO训练失败")
                response = input("是否继续评估? (y/n): ")
                if response.lower() != 'y':
                    return
        
        # 步骤5: 评估
        if not self.run_script("5_evaluation.py", "Step 5: 综合评估"):
            print("\n⚠️  评估失败")
        
        # 总结
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        
        print("\n" + "="*70)
        print("🎉 所有步骤完成!")
        print("="*70)
        print(f"\n⏱️  总耗时: {hours}小时 {minutes}分钟")
        print(f"\n📁 结果保存在:")
        print(f"   - 模型: {self.base_dir / 'models'}")
        print(f"   - 结果: {self.base_dir / 'results'}")
        print(f"\n📌 下一步:")
        print(f"   运行Demo: python scripts/6_demo_app.py")
        print(f"   或执行: python scripts/run_all.py --demo")
    
    def run_demo(self):
        """只运行Demo"""
        print("\n🚀 启动Demo应用...")
        
        demo_script = self.scripts_dir / "6_demo_app.py"
        
        if not demo_script.exists():
            print("❌ Demo脚本不存在")
            return
        
        subprocess.run(
            [sys.executable, str(demo_script)],
            cwd=str(self.base_dir)
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="中文情感分析项目 - 自动运行脚本")
    parser.add_argument("--skip-training", action="store_true", 
                       help="跳过训练步骤（仅运行数据准备和评估）")
    parser.add_argument("--demo", action="store_true", 
                       help="直接启动Demo应用")
    parser.add_argument("--quick", action="store_true",
                       help="快速模式：使用小数据集和少量epoch")
    
    args = parser.parse_args()
    
    runner = ProjectRunner()
    
    if args.demo:
        runner.run_demo()
    else:
        if args.quick:
            print("⚡ 快速模式：将使用较小的数据集和训练轮数")
            print("   （适合快速测试，结果可能不如完整训练）")
            time.sleep(2)
        
        runner.run_all(skip_training=args.skip_training)