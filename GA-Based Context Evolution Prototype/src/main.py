import os
import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown

from langgraph.graph import StateGraph, END

from src.models import Gene, AgentConfig
from src.state import AgentState
from src.environment import EnvironmentManager
from src.nodes import perception_node, agent_node, judge_node, should_continue

# 加载环境变量
load_dotenv()

# 从环境变量加载配置
INITIAL_ENERGY = int(os.getenv("INITIAL_ENERGY", "100"))

console = Console()  # Rich库的控制台输出对象，用于美化输出

def create_simulation_graph():
    """
    创建LangGraph状态机工作流
    逻辑：感知 -> 行动 -> 判断 -> (条件分支) -> 继续感知或结束
    目的：定义Agent在环境中交互的核心循环
    """
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("perception", perception_node)  # 感知节点：获取环境信息
    workflow.add_node("agent", agent_node)            # 行动节点：Agent生成响应
    workflow.add_node("judge", judge_node)             # 判断节点：评估行动有效性
    
    # 设置入口点
    workflow.set_entry_point("perception")
    
    # 添加边（连接节点）
    workflow.add_edge("perception", "agent")  # 感知完成后执行行动
    workflow.add_edge("agent", "judge")        # 行动完成后进行评估
    
    # 从判断节点添加条件边
    workflow.add_conditional_edges(
        "judge",
        should_continue,
        {
            "perception": "perception",  # 需要继续：回到感知节点
            "end": END                   # 结束：终止流程
        }
    )
    
    return workflow.compile()

def run_single_agent_simulation():
    """
    运行单个Agent的模拟实验
    逻辑：初始化环境 -> 创建Agent基因 -> 设置初始状态 -> 运行模拟循环
    目的：演示Agent在教程岛环境中的完整生存过程
    """
    console.print(Panel.fit("[bold green]基于遗传算法的上下文进化原型[/bold green]", subtitle="单Agent测试"))
    
    # 1. 设置环境
    env_manager = EnvironmentManager()
    try:
        scenario_steps = env_manager.load_scenario("tutorial_island")
        console.print(f"[blue]加载场景: 'tutorial_island' 包含 {len(scenario_steps)} 个步骤.[/blue]")
    except Exception as e:
        console.print(f"[bold red]加载场景错误: {e}[/bold red]")
        return

    # 2. 创建Agent（基因）
    # MVP阶段：手动创建基因
    gene = Gene(
        identity="你是一位冒险的考古学家。你勇敢但谨慎。",
        strategy="我总是寻找环境中的细节。我会立即使用找到的物品。",
        memory=""
    )
    
    agent_config = AgentConfig(gene=gene, initial_energy=INITIAL_ENERGY)
    
    # 3. 初始化状态
    initial_state = AgentState(
        agent_id=agent_config.id,
        gene=gene,
        generation=1,
        energy=agent_config.initial_energy,
        is_alive=True,
        cause_of_death=None,
        current_step_index=0,
        environment_steps=scenario_steps,
        current_environment_content="",
        messages=[],
        last_action_valid=False,
        feedback="",
        solved_steps=[]
    )
    
    # 4. 运行模拟
    app = create_simulation_graph()
    
    console.print("[yellow]开始模拟...[/yellow]")
    
    # 我们遍历流以打印更新
    step_count = 0
    max_steps = 20 # 安全中断机制
    
    current_state = initial_state
    
    # LangGraph流返回事件
    for event in app.stream(initial_state):
        step_count += 1
        
        for node_name, node_state in event.items():
            current_state.update(node_state) # 本地跟踪最新状态（如果需要）
            
            if node_name == "agent":
                # 打印Agent的行动
                last_msg = node_state["messages"][-1]
                console.print(Panel(
                    Markdown(last_msg.content), 
                    title=f"Agent (能量: {node_state.get('energy', '?')})", 
                    border_style="green"
                ))
            
            elif node_name == "judge":
                # 打印裁判的反馈
                feedback = node_state.get("feedback", "")
                reward = node_state.get("energy", 0) - current_state.get("energy", 0) # 由于状态更新逻辑，差异计算很棘手
                # 实际上node_state包含更新后的字段
                # 所以node_state['energy']是新的能量值
                
                is_solved = node_state.get("last_action_valid", False)
                color = "blue" if is_solved else "red"
                title = "裁判: 成功" if is_solved else "裁判: 失败"
                
                console.print(Panel(
                    Text(feedback, style="white"),
                    title=title,
                    border_style=color
                ))

            elif node_name == "perception":
                # 打印环境
                content = node_state.get("current_environment_content", "")
                console.print(Panel(
                    Markdown(content),
                    title=f"环境 (步骤 {current_state.get('current_step_index', 0)})",
                    border_style="yellow"
                ))
                
        if not current_state.get("is_alive", True):
            console.print(f"[bold red]Agent死亡: {current_state.get('cause_of_death', '未知')}[/bold red]")
            break
            
        if step_count > max_steps:
            console.print("[bold red]模拟停止（达到最大步数）[/bold red]")
            break

    console.print("[bold green]模拟结束.[/bold green]")

if __name__ == "__main__":
    run_single_agent_simulation()
