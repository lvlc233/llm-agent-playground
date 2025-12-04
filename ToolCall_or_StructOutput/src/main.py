import argparse
import sys
import os
from datetime import datetime

# Add project root to sys.path to allow running this script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.experiments import exp1_token_consumption
from src.experiments import exp2_multi_tool
from src.experiments import exp3_layered_architecture
from src.utils.llm_factory import get_available_models

def main():
    parser = argparse.ArgumentParser(
        description="运行工具调用与结构化输出对比实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
实验说明：
  实验1：Token消耗与基础提取对比
    - 对比工具调用和结构化输出在token消耗上的差异
    - 测试用户资料提取任务的性能

  实验2：多工具批处理性能对比
    - 测试多城市天气查询的批处理能力
    - 对比两种方法的并发处理效果

  实验3：分层架构性能对比
    - 测试分层选择架构vs标准工具调用的性能
    - 验证上下文污染减少的效果

输出说明：
  - 控制台显示中文实验结果和统计信息
  - 自动生成CSV文件保存详细数据
  - CSV文件保存在"实验结果"目录下
        """
    )
    parser.add_argument(
        "--exp",
        type=str,
        choices=["1", "2", "3", "all"],
        default="all",
        help="选择要运行的实验（1=Token消耗，2=多工具批处理，3=分层架构，all=全部）"
    )

    args = parser.parse_args()

    # 显示欢迎信息
    print("="*60)
    print("工具调用 vs 结构化输出 对比实验平台")
    print("="*60)
    print(f"运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"实验模式：{'全部实验' if args.exp == 'all' else f'实验{args.exp}'}")
    print("="*60)

    # 运行实验 - 支持多模型测试
    available_models = get_available_models()
    if not available_models:
        print("警告：未找到配置的模型，请在.env文件中设置MODEL_NAMES")
        available_models = ["Qwen/Qwen3-30B-A3B-Instruct-2507"]  # 默认模型

    print(f"可用模型: {', '.join(available_models)}")

    if args.exp in ["1", "all"]:
        print("\n" + "="*50)
        print("开始运行实验1：Token消耗与基础提取对比")
        print("="*50)
        print(f"测试模型数量: {len(available_models)}")
        exp1_token_consumption.run_experiment_multi_model(available_models)

    if args.exp in ["2", "all"]:
        print("\n" + "="*50)
        print("开始运行实验2：多工具批处理性能对比")
        print("="*50)
        print(f"测试模型数量: {len(available_models)}")
        exp2_multi_tool.run_experiment_multi_model(available_models)

    if args.exp in ["3", "all"]:
        print("\n" + "="*50)
        print("开始运行实验3：分层架构性能对比")
        print("="*50)
        print(f"测试模型数量: {len(available_models)}")
        # 实验3暂时使用单模型，后续再优化
        from src.utils.llm_factory import get_llm
        llm = get_llm()
        exp3_layered_architecture.run_experiment_with_llm(llm)

    # 显示完成信息
    print("\n" + "="*60)
    print("所有实验运行完成！")
    print("实验结果已保存到CSV文件，请查看'实验结果'目录")
    print("="*60)

if __name__ == "__main__":
    main()
