#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版状态驱动Demo
展示状态驱动vs传统ReAct的核心差异
"""

import time
import asyncio
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class ExecutionMode(Enum):
    """执行模式"""
    TRADITIONAL_REACT = "traditional_react"
    STATE_DRIVEN = "state_driven"


@dataclass
class ExecutionResult:
    """执行结果"""
    mode: ExecutionMode
    total_time: float
    steps: List[Dict[str, Any]]
    final_result: str
    concurrent_operations: int


class TraditionalReActAgent:
    """传统ReAct模式Agent"""
    
    def execute(self, user_input: str) -> ExecutionResult:
        """串行执行：Think -> Act -> Observe"""
        print(f"🔄 [传统ReAct] 开始处理: {user_input}")
        start_time = time.time()
        steps = []
        
        # Step 1: Think (思考)
        print("  🧠 思考阶段...")
        time.sleep(1.0)  # 模拟思考时间
        thought = f"分析用户需求: {user_input}"
        steps.append({
            "step": "think",
            "content": thought,
            "timestamp": datetime.now().isoformat(),
            "duration": 1.0
        })
        
        # Step 2: Act (行动)
        print("  ⚡ 执行阶段...")
        time.sleep(1.5)  # 模拟执行时间
        action_result = f"执行搜索: {user_input}"
        steps.append({
            "step": "act",
            "content": action_result,
            "timestamp": datetime.now().isoformat(),
            "duration": 1.5
        })
        
        # Step 3: Observe (观察)
        print("  👀 观察阶段...")
        time.sleep(0.5)  # 模拟观察时间
        observation = "获得搜索结果并分析"
        steps.append({
            "step": "observe",
            "content": observation,
            "timestamp": datetime.now().isoformat(),
            "duration": 0.5
        })
        
        total_time = time.time() - start_time
        final_result = f"传统模式完成: {thought} -> {action_result} -> {observation}"
        
        return ExecutionResult(
            mode=ExecutionMode.TRADITIONAL_REACT,
            total_time=total_time,
            steps=steps,
            final_result=final_result,
            concurrent_operations=0
        )


class StateDrivenAgent:
    """状态驱动模式Agent"""
    
    async def execute(self, user_input: str) -> ExecutionResult:
        """并发执行：Think || Act || Observe"""
        print(f"🚀 [状态驱动] 开始处理: {user_input}")
        start_time = time.time()
        steps = []
        
        # 创建并发任务
        think_task = asyncio.create_task(self._think_async(user_input))
        act_task = asyncio.create_task(self._act_async(user_input))
        observe_task = asyncio.create_task(self._observe_async())
        
        # 并发执行所有任务
        print("  🔄 并发执行: 思考 || 行动 || 观察")
        
        # 等待所有任务完成
        think_result, act_result, observe_result = await asyncio.gather(
            think_task, act_task, observe_task
        )
        
        steps.extend([think_result, act_result, observe_result])
        
        total_time = time.time() - start_time
        final_result = f"状态驱动完成: 并发处理 {len(steps)} 个操作"
        
        return ExecutionResult(
            mode=ExecutionMode.STATE_DRIVEN,
            total_time=total_time,
            steps=steps,
            final_result=final_result,
            concurrent_operations=3
        )
    
    async def _think_async(self, user_input: str) -> Dict[str, Any]:
        """异步思考"""
        await asyncio.sleep(1.0)  # 模拟思考时间
        return {
            "step": "think",
            "content": f"并发思考: {user_input}",
            "timestamp": datetime.now().isoformat(),
            "duration": 1.0
        }
    
    async def _act_async(self, user_input: str) -> Dict[str, Any]:
        """异步行动"""
        await asyncio.sleep(1.5)  # 模拟执行时间
        return {
            "step": "act",
            "content": f"并发执行: {user_input}",
            "timestamp": datetime.now().isoformat(),
            "duration": 1.5
        }
    
    async def _observe_async(self) -> Dict[str, Any]:
        """异步观察"""
        await asyncio.sleep(0.5)  # 模拟观察时间
        return {
            "step": "observe",
            "content": "并发观察和分析",
            "timestamp": datetime.now().isoformat(),
            "duration": 0.5
        }


class PerformanceComparator:
    """性能对比器"""
    
    def __init__(self):
        self.traditional_agent = TraditionalReActAgent()
        self.state_driven_agent = StateDrivenAgent()
    
    async def compare(self, user_input: str) -> Dict[str, Any]:
        """对比两种模式的性能"""
        print(f"🎯 开始性能对比: {user_input}")
        print("=" * 60)
        
        # 测试传统模式
        print("\n📊 测试传统ReAct模式:")
        traditional_result = self.traditional_agent.execute(user_input)
        
        print("\n📊 测试状态驱动模式:")
        state_driven_result = await self.state_driven_agent.execute(user_input)
        
        # 计算性能提升
        time_saved = traditional_result.total_time - state_driven_result.total_time
        improvement_percent = (time_saved / traditional_result.total_time) * 100
        
        comparison = {
            "traditional": {
                "time": traditional_result.total_time,
                "steps": len(traditional_result.steps),
                "concurrent_ops": traditional_result.concurrent_operations,
                "result": traditional_result.final_result
            },
            "state_driven": {
                "time": state_driven_result.total_time,
                "steps": len(state_driven_result.steps),
                "concurrent_ops": state_driven_result.concurrent_operations,
                "result": state_driven_result.final_result
            },
            "improvement": {
                "time_saved": time_saved,
                "percent_faster": improvement_percent,
                "concurrency_gain": state_driven_result.concurrent_operations
            }
        }
        
        return comparison
    
    def print_comparison(self, comparison: Dict[str, Any]):
        """打印对比结果"""
        print("\n" + "=" * 60)
        print("📈 性能对比结果:")
        print("-" * 40)
        
        trad = comparison["traditional"]
        state = comparison["state_driven"]
        improve = comparison["improvement"]
        
        print(f"传统ReAct模式:")
        print(f"  ⏱️  执行时间: {trad['time']:.2f}秒")
        print(f"  📝 执行步骤: {trad['steps']}个 (串行)")
        print(f"  🔄 并发操作: {trad['concurrent_ops']}个")
        
        print(f"\n状态驱动模式:")
        print(f"  ⏱️  执行时间: {state['time']:.2f}秒")
        print(f"  📝 执行步骤: {state['steps']}个 (并行)")
        print(f"  🔄 并发操作: {state['concurrent_ops']}个")
        
        print(f"\n🚀 性能提升:")
        print(f"  ⚡ 节省时间: {improve['time_saved']:.2f}秒")
        print(f"  📊 速度提升: {improve['percent_faster']:.1f}%")
        print(f"  🎯 并发优势: {improve['concurrency_gain']}个并发操作")
        
        print("\n💡 关键优势:")
        print("  • 思考与执行并发进行，减少等待时间")
        print("  • 状态驱动解耦，提高系统响应性")
        print("  • 更接近人类认知模式：边思考边行动")
        print("  • 异步处理提高资源利用率")


async def main():
    """主演示函数"""
    print("🎯 状态驱动 vs 传统ReAct 性能对比Demo")
    print("展示并发思考与执行的优势")
    
    comparator = PerformanceComparator()
    
    test_cases = [
        "搜索人工智能最新发展",
        "分析市场趋势数据",
        "处理复杂查询请求"
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 测试用例 {i}/{len(test_cases)}: {test_case}")
        
        comparison = await comparator.compare(test_case)
        comparator.print_comparison(comparison)
        
        if i < len(test_cases):
            print("\n" + "=" * 80)
            await asyncio.sleep(1)
    
    print("\n✨ Demo完成! 状态驱动模式展示了显著的性能优势。")


if __name__ == "__main__":
    asyncio.run(main())