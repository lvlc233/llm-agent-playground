#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
感知型智能体Demo运行脚本
简化版本，便于快速测试
"""

import asyncio
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from perception_agent import PerceptionAgent
except ImportError as e:
    print("❌ 导入错误:")
    print(f"   {e}")
    print("\n💡 请先安装依赖:")
    print("   pip install -r requirements.txt")
    sys.exit(1)


async def simple_demo():
    """简化的演示函数"""
    print("🚀 启动感知型智能体Demo...\n")
    
    agent = PerceptionAgent()
    
    try:
        # 执行一次感知过程
        result = await agent.perceive()
        
        print("\n📊 感知结果摘要:")
        print(f"   外部刺激数量: {len(result['external_stimuli'])}")
        print(f"   内部信号数量: {len(result['internal_signals'])}")
        print(f"   有效感知数量: {len(result['perceptions'])}")
        print(f"   注意力焦点: {result['attention_focus']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        return False


def main():
    """主函数"""
    print("=" * 50)
    print("🧠 感知型智能体架构Demo")
    print("   基于LangGraph实现")
    print("=" * 50)
    
    try:
        success = asyncio.run(simple_demo())
        if success:
            print("\n✅ Demo运行成功!")
        else:
            print("\n❌ Demo运行失败!")
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断运行")
    except Exception as e:
        print(f"\n💥 未预期的错误: {e}")


if __name__ == "__main__":
    main()