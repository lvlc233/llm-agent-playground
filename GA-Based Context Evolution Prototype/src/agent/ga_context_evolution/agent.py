from langgraph.graph import StateGraph, END
from typing import Literal

from .schema import EvolutionState
from .nodes import (
    target_profiling_node,
    gene_extraction_node,
    population_sampling_node,
    context_generation_node,
    fitness_evaluation_node,
    evolution_analysis_node
)
from .config import default_config

def should_continue(state: EvolutionState) -> Literal["population_sampling_node", END]:
    """检查进化是否应继续的条件。"""
    current_round = state.get("generation_round", 0)
    
    # 检查最大轮数
    if current_round >= state.get("max_rounds", default_config.max_generations):
        print(">>> 达到最大代数。停止。")
        return END
    
    # 检查收敛性（可选，如果我们跟踪最大分数历史）
    # 目前仅使用简单的轮数限制。
    
    return "population_sampling_node"

# 构建图
workflow = StateGraph(EvolutionState)

# 添加节点
workflow.add_node("target_profiling_node", target_profiling_node)
workflow.add_node("gene_extraction_node", gene_extraction_node)
workflow.add_node("population_sampling_node", population_sampling_node)
workflow.add_node("context_generation_node", context_generation_node)
workflow.add_node("fitness_evaluation_node", fitness_evaluation_node)
workflow.add_node("evolution_analysis_node", evolution_analysis_node)

# 设置入口点
workflow.set_entry_point("target_profiling_node")

# 添加边
workflow.add_edge("target_profiling_node", "gene_extraction_node")
workflow.add_edge("gene_extraction_node", "population_sampling_node")
workflow.add_edge("population_sampling_node", "context_generation_node")
workflow.add_edge("context_generation_node", "fitness_evaluation_node")
workflow.add_edge("fitness_evaluation_node", "evolution_analysis_node")

# 从分析节点回到采样节点或结束的条件边
workflow.add_conditional_edges(
    "evolution_analysis_node",
    should_continue,
    {
        "population_sampling_node": "population_sampling_node",
        END: END
    }
)

# 编译
graph = workflow.compile()
