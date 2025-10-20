#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
感知型智能体架构Demo - 基于LangGraph实现
实现感知层的四个核心组件：感受器、信号处理器、感觉生成器、注意力调控器
"""

import asyncio
import time
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


class StimulusType(Enum):
    """外部刺激类型"""
    VISUAL = "visual"      # 视觉刺激
    AUDIO = "audio"        # 听觉刺激
    INTERNAL = "internal"  # 内部状态刺激


@dataclass
class ExternalStimulus:
    """外部刺激数据结构"""
    stimulus_type: StimulusType
    intensity: float  # 刺激强度 0-1
    content: str     # 刺激内容
    timestamp: float


@dataclass
class InternalSignal:
    """内部信号数据结构"""
    signal_id: str
    processed_content: str
    confidence: float  # 信号置信度 0-1
    priority: float   # 信号优先级 0-1
    timestamp: float


@dataclass
class Perception:
    """感知结果数据结构"""
    perception_id: str
    content: str
    attention_weight: float  # 注意力权重 0-1
    timestamp: float


class AgentState(TypedDict):
    """智能体状态"""
    external_stimuli: List[ExternalStimulus]
    internal_signals: List[InternalSignal]
    perceptions: List[Perception]
    attention_focus: str  # 当前注意力焦点
    total_resources: float  # 总计算资源
    messages: Annotated[List, add_messages]


class PerceptionAgent:
    """感知型智能体"""
    
    def __init__(self):
        self.graph = self._build_graph()
        self.attention_weights = {
            StimulusType.VISUAL: 0.4,
            StimulusType.AUDIO: 0.3,
            StimulusType.INTERNAL: 0.3
        }
    
    def _build_graph(self) -> StateGraph:
        """构建感知处理图"""
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("receptor", self._receptor_node)
        workflow.add_node("signal_processor", self._signal_processor_node)
        workflow.add_node("sensation_generator", self._sensation_generator_node)
        workflow.add_node("attention_controller", self._attention_controller_node)
        
        # 定义流程
        workflow.set_entry_point("receptor")
        workflow.add_edge("receptor", "signal_processor")
        workflow.add_edge("signal_processor", "attention_controller")
        workflow.add_edge("attention_controller", "sensation_generator")
        workflow.add_edge("sensation_generator", END)
        
        return workflow.compile()
    
    async def _receptor_node(self, state: AgentState) -> AgentState:
        """感受器节点 - 接收外部刺激并转换为内部信号"""
        print("🔍 感受器工作中...")
        
        # 模拟接收外部刺激
        stimuli = await self._simulate_external_stimuli()
        state["external_stimuli"] = stimuli
        
        print(f"接收到 {len(stimuli)} 个外部刺激")
        for stimulus in stimuli:
            print(f"  - {stimulus.stimulus_type.value}: {stimulus.content} (强度: {stimulus.intensity:.2f})")
        
        return state
    
    async def _signal_processor_node(self, state: AgentState) -> AgentState:
        """信号处理器节点 - 将外部刺激转换为内部信号"""
        print("\n⚙️ 信号处理器工作中...")
        
        internal_signals = []
        for stimulus in state["external_stimuli"]:
            # 模拟信号处理过程
            processed_content = f"已处理: {stimulus.content}"
            confidence = min(stimulus.intensity + random.uniform(0, 0.3), 1.0)
            priority = stimulus.intensity * random.uniform(0.8, 1.2)
            
            signal = InternalSignal(
                signal_id=f"signal_{len(internal_signals)}",
                processed_content=processed_content,
                confidence=confidence,
                priority=min(priority, 1.0),
                timestamp=time.time()
            )
            internal_signals.append(signal)
        
        state["internal_signals"] = internal_signals
        print(f"生成了 {len(internal_signals)} 个内部信号")
        
        return state
    
    async def _attention_controller_node(self, state: AgentState) -> AgentState:
        """注意力调控器节点 - 根据资源限制调节注意力分配"""
        print("\n🎯 注意力调控器工作中...")
        
        # 模拟资源限制 (总注意力资源为1.0)
        total_resources = 1.0
        state["total_resources"] = total_resources
        
        # 根据信号优先级和当前注意力权重分配资源
        signals = state["internal_signals"]
        if not signals:
            return state
        
        # 计算注意力分配
        total_priority = sum(signal.priority for signal in signals)
        if total_priority > 0:
            for signal in signals:
                # 注意力权重 = (信号优先级 / 总优先级) * 总资源
                attention_weight = (signal.priority / total_priority) * total_resources
                signal.attention_weight = attention_weight
        
        # 确定当前注意力焦点
        focused_signal = max(signals, key=lambda s: getattr(s, 'attention_weight', 0))
        state["attention_focus"] = focused_signal.signal_id
        
        print(f"注意力焦点: {focused_signal.signal_id}")
        print("注意力分配:")
        for signal in signals:
            weight = getattr(signal, 'attention_weight', 0)
            print(f"  - {signal.signal_id}: {weight:.3f}")
        
        return state
    
    async def _sensation_generator_node(self, state: AgentState) -> AgentState:
        """感觉生成器节点 - 基于注意力权重生成最终感知"""
        print("\n✨ 感觉生成器工作中...")
        
        perceptions = []
        for signal in state["internal_signals"]:
            attention_weight = getattr(signal, 'attention_weight', 0)
            
            # 只有注意力权重超过阈值的信号才会生成感知
            if attention_weight > 0.1:  # 注意力阈值
                perception = Perception(
                    perception_id=f"perception_{len(perceptions)}",
                    content=f"感知到: {signal.processed_content}",
                    attention_weight=attention_weight,
                    timestamp=time.time()
                )
                perceptions.append(perception)
        
        state["perceptions"] = perceptions
        
        print(f"生成了 {len(perceptions)} 个感知:")
        for perception in perceptions:
            print(f"  - {perception.content} (注意力权重: {perception.attention_weight:.3f})")
        
        return state
    
    async def _simulate_external_stimuli(self) -> List[ExternalStimulus]:
        """模拟外部刺激"""
        stimuli = [
            ExternalStimulus(
                stimulus_type=StimulusType.VISUAL,
                intensity=random.uniform(0.3, 0.9),
                content="检测到移动物体",
                timestamp=time.time()
            ),
            ExternalStimulus(
                stimulus_type=StimulusType.AUDIO,
                intensity=random.uniform(0.2, 0.8),
                content="听到背景音乐",
                timestamp=time.time()
            ),
            ExternalStimulus(
                stimulus_type=StimulusType.INTERNAL,
                intensity=random.uniform(0.4, 0.7),
                content="系统状态正常",
                timestamp=time.time()
            )
        ]
        
        # 随机选择1-3个刺激
        return random.sample(stimuli, random.randint(1, 3))
    
    async def perceive(self) -> Dict[str, Any]:
        """执行一次完整的感知过程"""
        print("🤖 感知型智能体开始工作...\n")
        
        initial_state = AgentState(
            external_stimuli=[],
            internal_signals=[],
            perceptions=[],
            attention_focus="",
            total_resources=1.0,
            messages=[]
        )
        
        # 执行感知流程
        result = await self.graph.ainvoke(initial_state)
        
        print("\n🎉 感知过程完成!")
        print(f"最终生成 {len(result['perceptions'])} 个有效感知")
        print(f"当前注意力焦点: {result['attention_focus']}")
        
        return result


async def main():
    """主函数 - 演示感知型智能体"""
    agent = PerceptionAgent()
    
    print("=" * 60)
    print("感知型智能体架构Demo")
    print("基于LangGraph实现的感知层组件")
    print("=" * 60)
    
    # 执行多轮感知
    for round_num in range(3):
        print(f"\n🔄 第 {round_num + 1} 轮感知:")
        print("-" * 40)
        
        result = await agent.perceive()
        
        # 短暂休息
        await asyncio.sleep(1)
    
    print("\n✅ Demo演示完成!")


if __name__ == "__main__":
    asyncio.run(main())