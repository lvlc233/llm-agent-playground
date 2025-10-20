#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
状态驱动Agent Demo启动器
提供交互式菜单来运行不同的演示
"""

import os
import sys
import asyncio
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        'langgraph',
        'langchain',
        'langchain_core',
        'asyncio'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n请运行以下命令安装依赖:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖已安装")
    return True


def check_environment():
    """检查环境配置"""
    print("🔍 检查环境配置...")
    
    # 检查OpenAI API Key（可选）
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("⚠️  未设置OPENAI_API_KEY环境变量")
        print("   如果要使用OpenAI模型，请设置此变量")
        print("   或者可以使用本地模拟模式运行demo")
    else:
        print("✅ OPENAI_API_KEY已设置")
    
    return True


def print_menu():
    """打印菜单"""
    print("\n" + "=" * 60)
    print("🎯 状态驱动Agent模式Demo")
    print("=" * 60)
    print("请选择要运行的演示:")
    print()
    print("1. 🚀 简化版性能对比Demo")
    print("   - 展示传统ReAct vs 状态驱动的性能差异")
    print("   - 无需API Key，使用模拟数据")
    print("   - 推荐首次体验")
    print()
    print("2. 🔧 完整版状态驱动Agent")
    print("   - 完整的LangGraph实现")
    print("   - 需要OpenAI API Key")
    print("   - 展示真实的LLM交互")
    print()
    print("3. 📚 查看项目文档")
    print("   - 打开README文档")
    print()
    print("4. 🧪 运行测试用例")
    print("   - 批量测试不同场景")
    print()
    print("0. 退出")
    print("=" * 60)


async def run_simple_demo():
    """运行简化版Demo"""
    print("\n🚀 启动简化版性能对比Demo...")
    try:
        from simple_state_demo import main
        await main()
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        print("请检查simple_state_demo.py文件是否存在")


async def run_full_demo():
    """运行完整版Demo"""
    print("\n🔧 启动完整版状态驱动Agent...")
    

    
    try:
        from state_driven_agent_demo import main
        await main()
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        print("请检查state_driven_agent_demo.py文件是否存在")


def view_documentation():
    """查看文档"""
    print("\n📚 项目文档:")
    print("-" * 40)
    
    readme_path = current_dir / "demo_README.md"
    if readme_path.exists():
        print(f"📄 主要文档: {readme_path}")
        
        # 尝试在默认编辑器中打开
        try:
            if sys.platform.startswith('win'):
                os.startfile(readme_path)
            elif sys.platform.startswith('darwin'):
                os.system(f'open "{readme_path}"')
            else:
                os.system(f'xdg-open "{readme_path}"')
            print("✅ 已在默认编辑器中打开文档")
        except:
            print("⚠️  无法自动打开，请手动查看文件")
    else:
        print("❌ 未找到README文档")
    
    # 显示其他相关文件
    other_docs = [
        "状态驱动的工具调用模式demo/README.md",
        "requirements.txt"
    ]
    
    print("\n📋 其他相关文件:")
    for doc in other_docs:
        doc_path = current_dir / doc
        if doc_path.exists():
            print(f"  ✅ {doc}")
        else:
            print(f"  ❌ {doc} (未找到)")


async def run_test_cases():
    """运行测试用例"""
    print("\n🧪 运行测试用例...")
    
    test_cases = [
        "搜索人工智能的最新发展趋势",
        "分析当前股市的技术指标",
        "计算复合增长率公式",
        "总结机器学习的核心概念",
        "比较不同编程语言的特点"
    ]
    
    print(f"准备测试 {len(test_cases)} 个用例...")
    
    # 选择运行模式
    print("\n选择测试模式:")
    print("1. 简化模式 (无需API Key)")
    print("2. 完整模式 (需要API Key)")
    
    choice = input("请选择 (1/2): ").strip()
    
    if choice == "1":
        # 简化模式测试
        try:
            from simple_state_demo import PerformanceComparator
            comparator = PerformanceComparator()
            
            for i, test_case in enumerate(test_cases, 1):
                print(f"\n🧪 测试用例 {i}/{len(test_cases)}: {test_case}")
                comparison = await comparator.compare(test_case)
                comparator.print_comparison(comparison)
                
                if i < len(test_cases):
                    await asyncio.sleep(1)
                    
        except Exception as e:
            print(f"❌ 测试出错: {e}")
            
    elif choice == "2":
        # 完整模式测试
        if not os.getenv('OPENAI_API_KEY'):
            print("❌ 完整模式需要设置OPENAI_API_KEY环境变量")
            return
            
        try:
            from state_driven_agent_demo import StateBasedAgent
            agent = StateBasedAgent()
            
            for i, test_case in enumerate(test_cases, 1):
                print(f"\n🧪 测试用例 {i}/{len(test_cases)}: {test_case}")
                result = await agent.run(test_case)
                print(f"✅ 完成: {result['response'][:100]}...")
                
                if i < len(test_cases):
                    await asyncio.sleep(2)
                    
        except Exception as e:
            print(f"❌ 测试出错: {e}")
    else:
        print("❌ 无效选择")


async def main():
    """主函数"""
    print("🎯 状态驱动Agent模式Demo启动器")
    
    # 检查环境
    if not check_dependencies():
        return
    
    check_environment()
    
    while True:
        print_menu()
        choice = input("请选择 (0-4): ").strip()
        
        if choice == "0":
            print("👋 感谢使用！")
            break
        elif choice == "1":
            await run_simple_demo()
        elif choice == "2":
            await run_full_demo()
        elif choice == "3":
            view_documentation()
        elif choice == "4":
            await run_test_cases()
        else:
            print("❌ 无效选择，请重新输入")
        
        input("\n按Enter键继续...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 用户中断，退出程序")
    except Exception as e:
        print(f"\n❌ 程序出错: {e}")
        print("请检查环境配置和依赖安装")