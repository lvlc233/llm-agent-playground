#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
状态驱动的工具调用模式Demo
基于LangGraph实现思考与执行并发的Agent架构
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


class AgentState(Enum):
    """Agent状态枚举"""
    IDLE = "idle"
    THINKING = "thinking"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    ERROR = "error"


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """任务数据结构"""
    id: str
    name: str
    description: str
    tool_name: str
    params: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class ThoughtProcess:
    """思考过程数据结构"""
    id: str
    content: str
    reasoning: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    related_tasks: List[str] = field(default_factory=list)


class StateBasedAgentState(TypedDict):


    messages: Annotated[List[BaseMessage], add_messages]
    agent_state: AgentState
    current_thought: Optional[ThoughtProcess]
    task_queue: List[Task]
    active_tasks: Dict[str, Task]
    completed_tasks: List[Task]
    context: Dict[str, Any]
    execution_log: List[Dict[str, Any]]


class StateBasedAgent:
    """状态驱动的Agent实现"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"):
        self.llm = ChatOpenAI(base_url="https://api.siliconflow.cn/v1",api_key="sk-lbzsbznhrwackhgeybltsfsghkkllfsafzgttvpvnlgpajol",model=model_name, temperature=0.7)
        ### 线程池执行器
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """构建状态图"""
        workflow = StateGraph(StateBasedAgentState)
        
        # 添加节点
        workflow.add_node("think", self._think_node)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("execute", self._execute_node)
        workflow.add_node("reflect", self._reflect_node)
        workflow.add_node("state_manager", self._state_manager_node)
        
        # 设置入口点
        workflow.set_entry_point("think")
        
        # 添加边
        workflow.add_edge("think", "state_manager")
        workflow.add_edge("plan", "state_manager")
        workflow.add_edge("execute", "state_manager")
        workflow.add_edge("reflect", "state_manager")
        
        # 状态管理器决定下一步
        workflow.add_conditional_edges(
            "state_manager",
            self._route_next_action,
            {
                "think": "think",
                "plan": "plan",
                "execute": "execute",
                "reflect": "reflect",
                "end": END
            }
        )
        
        return workflow.compile()
    
    ### 记录思考内容和推理内容
    async def _think_node(self, state: StateBasedAgentState) -> StateBasedAgentState:
        """思考节点 - 并发执行思考过程"""
        print(f"🧠 [THINKING] Agent开始思考...")
        
        # 获取最新消息
        last_message = state["messages"][-1] if state["messages"] else None
        
        if isinstance(last_message, HumanMessage):
            # 异步思考过程
            ### 提交一个任务到线程池中,这个任务是_generate_thought,并想_generate_thought函数传入后面的两个参数"
            thought_future = self.executor.submit(
                self._generate_thought, 
                last_message.content, 
                state["context"]
            )
            
            # 模拟思考需要时间，但不阻塞
            await asyncio.sleep(0.1)
            
            try:
                thought = thought_future.result(timeout=5)
                state["current_thought"] = thought
                state["agent_state"] = AgentState.PLANNING
                
                # 记录思考过程
                state["execution_log"].append({
                    "timestamp": datetime.now().isoformat(),
                    "action": "think",
                    "content": thought.content,
                    "reasoning": thought.reasoning
                })
                
                print(f"💭 思考内容: {thought.content}")
                print(f"🔍 推理过程: {thought.reasoning}")
                
            except Exception as e:
                print(f"❌ 思考过程出错: {e}")
                state["agent_state"] = AgentState.ERROR
        
        return state
    ### 调用模型进行思考任务
    def _generate_thought(self, user_input: str, context: Dict[str, Any]) -> ThoughtProcess:
        """生成思考过程"""
        prompt = f"""
        用户输入: {user_input}
        当前上下文: {json.dumps(context, ensure_ascii=False, indent=2)}
        
        请分析用户需求，生成思考过程。返回JSON格式：
        {{
            "content": "思考的主要内容",
            "reasoning": "推理过程",
            "confidence": 0.8,
            "related_tasks": ["可能需要的任务列表"]
        }}
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            thought_data = json.loads(response.content)
            return ThoughtProcess(
                id=f"thought_{int(time.time())}",
                content=thought_data["content"],
                reasoning=thought_data["reasoning"],
                confidence=thought_data["confidence"],
                related_tasks=thought_data.get("related_tasks", [])
            )
        except:
            return ThoughtProcess(
                id=f"thought_{int(time.time())}",
                content=f"分析用户需求: {user_input}",
                reasoning="基于用户输入进行基础分析",
                confidence=0.6
            )
    ### 基于思考结果生成任务
    async def _plan_node(self, state: StateBasedAgentState) -> StateBasedAgentState:
        """规划节点 - 基于思考结果制定执行计划"""
        print(f"📋 [PLANNING] Agent开始制定计划...")
        
        current_thought = state["current_thought"]
        if not current_thought:
            state["agent_state"] = AgentState.ERROR
            return state
        
        # 基于思考结果生成任务
        tasks = self._generate_tasks(current_thought, state["context"])
        
        # 将任务添加到队列
        state["task_queue"].extend(tasks)
        state["agent_state"] = AgentState.EXECUTING
        
        print(f"📝 生成了 {len(tasks)} 个任务:")
        for task in tasks:
            print(f"  - {task.name}: {task.description}")
        
        return state
    ### 基于思考结果生成任务或者说工具信息
    def _generate_tasks(self, thought: ThoughtProcess, context: Dict[str, Any]) -> List[Task]:
        """基于思考过程生成任务列表"""
        tasks = []
        
        # 示例：基于思考内容生成搜索任务
        if "搜索" in thought.content or "查找" in thought.content:
            tasks.append(Task(
                id=f"search_{int(time.time())}",
                name="网络搜索",
                description=f"搜索相关信息: {thought.content}",
                tool_name="search_tool",
                params={"query": thought.content}
            ))
        
        # 示例：基于思考内容生成计算任务
        if "计算" in thought.content or "数学" in thought.content:
            tasks.append(Task(
                id=f"calc_{int(time.time())}",
                name="数学计算",
                description=f"执行计算: {thought.content}",
                tool_name="calculator_tool",
                params={"expression": thought.content}
            ))
        
        # 默认分析任务
        if not tasks:
            tasks.append(Task(
                id=f"analyze_{int(time.time())}",
                name="内容分析",
                description=f"分析内容: {thought.content}",
                tool_name="analyze_tool",
                params={"content": thought.content}
            ))
        
        return tasks
    
    async def _execute_node(self, state: StateBasedAgentState) -> StateBasedAgentState:
        """执行节点 - 并发执行任务"""
        print(f"⚡ [EXECUTING] Agent开始执行任务...")
        
        # 从队列中取出任务并发执行
        pending_tasks = [task for task in state["task_queue"] if task.status == TaskStatus.PENDING]
        
        if not pending_tasks:
            state["agent_state"] = AgentState.REFLECTING
            return state
        
        # 并发执行任务（最多3个）
        execution_futures = []
        for task in pending_tasks[:3]:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            state["active_tasks"][task.id] = task
            
            future = self.executor.submit(self._execute_task, task)
            execution_futures.append((task.id, future))
        
        # 等待任务完成（非阻塞）
        await asyncio.sleep(0.1)
        
        # 检查已完成的任务
        completed_task_ids = []
        for task_id, future in execution_futures:
            if future.done():
                try:
                    result = future.result()
                    task = state["active_tasks"][task_id]
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    
                    state["completed_tasks"].append(task)
                    completed_task_ids.append(task_id)
                    
                    print(f"✅ 任务完成: {task.name} -> {result}")
                    
                except Exception as e:
                    task = state["active_tasks"][task_id]
                    task.error = str(e)
                    task.status = TaskStatus.FAILED
                    completed_task_ids.append(task_id)
                    
                    print(f"❌ 任务失败: {task.name} -> {e}")
        
        # 清理已完成的任务
        for task_id in completed_task_ids:
            if task_id in state["active_tasks"]:
                del state["active_tasks"][task_id]
            # 从队列中移除
            state["task_queue"] = [t for t in state["task_queue"] if t.id != task_id]
        
        # 决定下一步
        if not state["active_tasks"] and not state["task_queue"]:
            state["agent_state"] = AgentState.REFLECTING
        else:
            state["agent_state"] = AgentState.EXECUTING
        
        return state
    
    def _execute_task(self, task: Task) -> Any:
        """执行具体任务"""
        print(f"🔧 执行任务: {task.name}")
        
        # 模拟不同工具的执行
        if task.tool_name == "search_tool":
            time.sleep(1)  # 模拟网络延迟
            return f"搜索结果: 找到关于'{task.params['query']}'的相关信息"
        
        elif task.tool_name == "calculator_tool":
            time.sleep(0.5)  # 模拟计算时间
            return f"计算结果: {task.params['expression']} = 42"
        
        elif task.tool_name == "analyze_tool":
            time.sleep(0.8)  # 模拟分析时间
            return f"分析结果: '{task.params['content']}'包含重要信息"
        
        else:
            return f"未知工具: {task.tool_name}"
    
    async def _reflect_node(self, state: StateBasedAgentState) -> StateBasedAgentState:
        """反思节点 - 分析执行结果并生成回复"""
        print(f"🤔 [REFLECTING] Agent开始反思...")
        
        # 收集执行结果
        results = []
        for task in state["completed_tasks"]:
            if task.result:
                results.append(f"{task.name}: {task.result}")
        
        # 生成回复
        if results:
            response_content = "\n".join([
                "基于状态驱动的并发执行，我完成了以下任务：",
                *[f"• {result}" for result in results],
                "",
                f"思考过程: {state['current_thought'].content if state['current_thought'] else '无'}",
                f"执行时间: 并发处理，提高了效率"
            ])
        else:
            response_content = "任务执行完成，但没有获得具体结果。"
        
        # 添加AI回复消息
        ai_message = AIMessage(content=response_content)
        state["messages"].append(ai_message)
        state["agent_state"] = AgentState.IDLE
        
        print(f"💬 生成回复: {response_content[:100]}...")
        
        return state
    
    def _state_manager_node(self, state: StateBasedAgentState) -> StateBasedAgentState:
        """状态管理节点 - 协调各个状态之间的转换"""
        current_state = state["agent_state"]
        print(f"🎛️  [STATE_MANAGER] 当前状态: {current_state.value}")
        
        # 记录状态变化
        state["execution_log"].append({
            "timestamp": datetime.now().isoformat(),
            "action": "state_change",
            "from_state": current_state.value,
            "active_tasks": len(state["active_tasks"]),
            "pending_tasks": len([t for t in state["task_queue"] if t.status == TaskStatus.PENDING])
        })
        
        return state
    
    def _route_next_action(self, state: StateBasedAgentState) -> str:
        """路由下一个动作"""
        current_state = state["agent_state"]
        
        if current_state == AgentState.THINKING:
            return "plan"
        elif current_state == AgentState.PLANNING:
            return "execute"
        elif current_state == AgentState.EXECUTING:
            # 如果还有任务在执行或等待，继续执行
            if state["active_tasks"] or any(t.status == TaskStatus.PENDING for t in state["task_queue"]):
                return "execute"
            else:
                return "reflect"
        elif current_state == AgentState.REFLECTING:
            return "end"
        elif current_state == AgentState.IDLE:
            return "end"
        elif current_state == AgentState.ERROR:
            return "end"
        else:
            return "end"
    
    async def run(self, user_input: str) -> Dict[str, Any]:
        """运行Agent"""
        print(f"🚀 启动状态驱动Agent...")
        print(f"📝 用户输入: {user_input}")
        print("=" * 60)
        
        # 初始化状态
        initial_state = StateBasedAgentState(
            messages=[HumanMessage(content=user_input)],
            agent_state=AgentState.THINKING,
            current_thought=None,
            task_queue=[],
            active_tasks={},
            completed_tasks=[],
            context={"user_input": user_input, "start_time": datetime.now().isoformat()},
            execution_log=[]
        )
        
        # 执行状态图
        result = await self.graph.ainvoke(initial_state)
        
        print("=" * 60)
        print(f"✨ Agent执行完成!")
        
        return {
            "response": result["messages"][-1].content if result["messages"] else "无回复",
            "execution_log": result["execution_log"],
            "completed_tasks": len(result["completed_tasks"]),
            "total_time": (datetime.now() - datetime.fromisoformat(result["context"]["start_time"])).total_seconds()
        }


async def main():
    """主函数 - 演示状态驱动Agent"""
    print("🎯 状态驱动工具调用模式Demo")
    print("展示思考与执行并发的Agent架构")
    print("=" * 60)
    
    agent = StateBasedAgent()
    
    # 测试用例
    test_cases = [
        "帮我搜索人工智能的最新发展",
        "计算一下复杂的数学问题",
        "分析一下当前的技术趋势"
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n🧪 测试用例 {i}: {test_input}")
        print("-" * 40)
        
        start_time = time.time()
        result = await agent.run(test_input)
        end_time = time.time()
        
        print(f"\n📊 执行结果:")
        print(f"回复: {result['response']}")
        print(f"完成任务数: {result['completed_tasks']}")
        print(f"总耗时: {result['total_time']:.2f}秒")
        print(f"实际耗时: {end_time - start_time:.2f}秒")
        
        if i < len(test_cases):
            print("\n" + "=" * 60)
            await asyncio.sleep(1)


if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())