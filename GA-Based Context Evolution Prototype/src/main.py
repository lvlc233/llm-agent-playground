import os
import argparse
import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown

from langgraph.graph import END

from src.models import Gene, AgentConfig
from src.state import AgentState
from src.environment import EnvironmentManager
from src.simulation import create_simulation_graph
from src.evolution import EvolutionEngine

# 加载环境变量
load_dotenv()

# 从环境变量加载配置
INITIAL_ENERGY = int(os.getenv("INITIAL_ENERGY", "100"))

console = Console()

def run_single_agent_demo():
    """
    运行单个Agent的演示模式（详细UI）
    """
    console.print(Panel.fit("[bold green]基于遗传算法的上下文进化原型[/bold green]", subtitle="单Agent演示模式"))
    
    # 1. 设置环境
    env_manager = EnvironmentManager()
    try:
        scenario_steps = env_manager.load_scenario("tutorial_island")
        console.print(f"[blue]加载场景: 'tutorial_island' 包含 {len(scenario_steps)} 个步骤.[/blue]")
    except Exception as e:
        console.print(f"[bold red]加载场景错误: {e}[/bold red]")
        return

    # 2. 创建Agent（基因）
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
    
    step_count = 0
    max_steps = 20
    current_state = initial_state
    
    for event in app.stream(initial_state):
        step_count += 1
        
        for node_name, node_state in event.items():
            current_state.update(node_state)
            
            if node_name == "agent":
                last_msg = node_state["messages"][-1]
                console.print(Panel(
                    Markdown(last_msg.content), 
                    title=f"Agent (能量: {node_state.get('energy', '?')})", 
                    border_style="green"
                ))
            
            elif node_name == "judge":
                feedback = node_state.get("feedback", "")
                is_solved = node_state.get("last_action_valid", False)
                color = "blue" if is_solved else "red"
                title = "裁判: 成功" if is_solved else "裁判: 失败"
                
                console.print(Panel(
                    Text(feedback, style="white"),
                    title=title,
                    border_style=color
                ))

            elif node_name == "perception":
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

    console.print("[bold green]演示结束.[/bold green]")

def run_evolution_mode(generations: int, population: int, log_dir: str = None):
    """
    运行进化模式
    """
    console.print(Panel.fit("[bold magenta]启动进化引擎[/bold magenta]", subtitle=f"Gen: {generations}, Pop: {population}"))
    engine = EvolutionEngine(population_size=population, generations=generations, log_dir=log_dir)
    engine.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GA-Based Context Evolution Prototype")
    parser.add_argument("--mode", "-m", choices=["demo", "evo"], default="demo", help="运行模式: demo(单体演示) 或 evo(进化循环)")
    parser.add_argument("--generations", "-g", type=int, default=3, help="进化代数 (仅evo模式)")
    parser.add_argument("--population", "-p", type=int, default=4, help="种群大小 (仅evo模式)")
    parser.add_argument("--log-dir", "-l", type=str, default=None, help="日志保存目录 (仅evo模式)")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        run_single_agent_demo()
    elif args.mode == "evo":
        run_evolution_mode(args.generations, args.population, args.log_dir)
