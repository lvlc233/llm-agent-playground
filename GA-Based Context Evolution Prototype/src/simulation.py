import os
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.nodes import perception_node, agent_node, judge_node, should_continue
from src.models import AgentConfig
from src.environment import EnvironmentManager

def create_simulation_graph():
    """
    创建LangGraph状态机工作流
    逻辑：感知 -> 行动 -> 判断 -> (条件分支) -> 继续感知或结束
    目的：定义Agent在环境中交互的核心循环
    """
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("perception", perception_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("judge", judge_node)
    
    # 设置入口点
    workflow.set_entry_point("perception")
    
    # 添加边
    workflow.add_edge("perception", "agent")
    workflow.add_edge("agent", "judge")
    
    # 添加条件边
    workflow.add_conditional_edges(
        "judge",
        should_continue,
        {
            "perception": "perception",
            "end": END
        }
    )
    
    return workflow.compile()

def run_simulation(agent_config: AgentConfig, env_manager: EnvironmentManager, max_steps: int = 20):
    """
    运行单个Agent的模拟（无头模式/Headless）
    
    Args:
        agent_config: Agent配置
        env_manager: 环境管理器
        max_steps: 最大步数
        
    Returns:
        dict: 模拟结果，包含 fitness, solved_steps, final_energy 等
    """
    # 1. 加载场景
    try:
        scenario_steps = env_manager.load_scenario("tutorial_island")
    except Exception as e:
        return {
            "error": str(e),
            "fitness": 0,
            "solved_steps_count": 0
        }

    # 2. 初始化状态
    initial_state = AgentState(
        agent_id=agent_config.id,
        gene=agent_config.gene,
        generation=agent_config.generation,
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
    
    # 3. 运行模拟
    app = create_simulation_graph()
    
    current_state = initial_state
    step_count = 0
    
    # 收集日志以便分析
    # 结构化日志: List[Dict]
    # {
    #   "step": int,
    #   "type": "perception" | "agent" | "judge",
    #   "content": str,
    #   "metadata": dict
    # }
    logs = []
    
    for event in app.stream(initial_state):
        step_count += 1
        
        for node_name, node_state in event.items():
            current_state.update(node_state)
            
            if node_name == "perception":
                content = node_state.get("current_environment_content", "")
                logs.append({
                    "step": step_count,
                    "type": "perception",
                    "content": content,
                    "metadata": {"step_index": current_state.get("current_step_index")}
                })

            elif node_name == "agent":
                last_msg = node_state["messages"][-1]
                logs.append({
                    "step": step_count,
                    "type": "agent",
                    "content": last_msg.content,
                    "metadata": {"energy": node_state.get("energy")}
                })
                
            elif node_name == "judge":
                feedback = node_state.get("feedback", "")
                is_solved = node_state.get("last_action_valid", False)
                logs.append({
                    "step": step_count,
                    "type": "judge",
                    "content": feedback,
                    "metadata": {
                        "is_solved": is_solved,
                        "energy_reward": node_state.get("energy", 0) - (current_state.get("energy", 0) - node_state.get("energy", 0)) # 修正: 這裡計算比較複雜，暫存狀態快照
                    }
                })

        if not current_state.get("is_alive", True):
            break
            
        if step_count > max_steps:
            current_state["cause_of_death"] = "Timeout"
            break
    
    # 4. 计算适应度
    # 适应度公式：解决的步骤数 * 100 + 剩余能量 + (100 - 使用步数)
    solved_count = len(current_state.get("solved_steps", []))
    final_energy = current_state.get("energy", 0)
    
    fitness = (solved_count * 100) + final_energy
    if current_state.get("is_alive", True):
        fitness += 20 # 存活奖励
        
    return {
        "agent_id": agent_config.id,
        "generation": agent_config.generation,
        "fitness": fitness,
        "solved_steps_count": solved_count,
        "final_energy": final_energy,
        "is_alive": current_state.get("is_alive", True),
        "cause_of_death": current_state.get("cause_of_death"),
        "gene": agent_config.gene,
        "logs": logs
    }
