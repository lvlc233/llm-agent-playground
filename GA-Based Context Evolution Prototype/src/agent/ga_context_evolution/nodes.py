import asyncio
import json
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from src.utils.llm_factory import LLMFactory
from .schema import EvolutionState, EvaluationResult, TargetFeature, ContextChunkList
from .config import default_config
from .prompts import (
    TARGET_PROFILING_PROMPT,
    GENE_EXTRACTION_PROMPT,
    CONTEXT_GENERATION_PROMPT,
    REVERSE_PROFILING_PROMPT,
    EVOLUTION_ANALYSIS_PROMPT
)
from .utils import (
    parse_json_output,
    create_initial_chunks,
    select_and_crossover,
    update_gene_pool_stats
)


async def target_profiling_node(state: EvolutionState) -> Dict[str, Any]:
    """分析目标并创建理想的画像 (Analyzes the target and creates an ideal profile)."""
    decomposer_llm = LLMFactory.create(default_config.decomposer_model)
    decomposer_llm = decomposer_llm.with_structured_output(TargetFeature)
    print(f"--- [节点] 目标画像生成 (Target Profiling) ---", flush=True)
    print("    正在调用分解模型生成画像...", flush=True)
    messages = await TARGET_PROFILING_PROMPT.ainvoke({"target_description": state["target_description"]})
    response: TargetFeature = await decomposer_llm.ainvoke(messages)
    profile = response.features
    print("    画像生成完成。", flush=True)
    return {"ideal_context_profile": profile}

async def gene_extraction_node(state: EvolutionState) -> Dict[str, Any]:
    """将画像分解为初始块 (Decomposes the profile into initial chunks)."""
    decomposer_llm = LLMFactory.create(default_config.decomposer_model)
    decomposer_llm = decomposer_llm.with_structured_output(ContextChunkList)
    print(f"--- [节点] 基因提取 (Gene Extraction) ---", flush=True)
    print("    正在调用分解模型进行基因提取...", flush=True)
    chain = GENE_EXTRACTION_PROMPT | decomposer_llm 
    output: ContextChunkList = await chain.ainvoke({
        "ideal_context_profile": state["ideal_context_profile"],
        "target_description": state["target_description"],
        "num_chunks": default_config.num_initial_chunks
    })
    print("    基因提取完成。", flush=True)
    
    gene_pool = {}
    for feature, chunks in output.chunks.items():
        gene_pool[feature] = create_initial_chunks(chunks)
    return {
        "gene_pool": gene_pool,
        "generation_round": 0,
        "max_rounds": default_config.max_generations
    }

async def population_sampling_node(state: EvolutionState) -> Dict[str, Any]:
    """创建新一代染色体 (Creates a new generation of chromosomes)."""
    current_round = state.get("generation_round", 0) + 1
    print(f"--- [节点] 种群采样 (Population Sampling) - 第 {current_round} 轮 ---", flush=True)
    
    chromosomes = select_and_crossover(
        state["gene_pool"], 
        state["ideal_context_profile"],
        default_config.initial_population_size
    )
    
    return {
        "chromosomes": chromosomes,
        "generation_round": current_round
    }

async def context_generation_node(state: EvolutionState) -> Dict[str, Any]:
    """并行生成所有染色体的上下文 (Generates contexts for all chromosomes in parallel)."""
    generator_llm = LLMFactory.create(default_config.generator_model)
    
    print(f"--- [节点] 上下文生成 (Context Generation) ---", flush=True)
    chromosomes = state["chromosomes"]
    
    # 扁平化基因映射: chunk_id -> content
    gene_map = {}
    for chunks in state["gene_pool"].values():
        for chunk in chunks:
            gene_map[chunk.id] = chunk.content
    
    async def generate_single(chromosome):
        print(f"    正在为染色体 {chromosome.id} 生成上下文...", flush=True)
        # 从块 ID 重构上下文字符串
        context_content = "\n\n".join(
            [gene_map.get(cid, "") for cid in chromosome.chunk_ids]
        )
        
        # 调用 LLM
        messages = await CONTEXT_GENERATION_PROMPT.ainvoke({
            "context_content": context_content,
            "target_description": state["target_description"]
        })
        
        response = await generator_llm.ainvoke(messages)
        print(f"    染色体 {chromosome.id} 的上下文生成完毕。", flush=True)
        return {
            "chromosome_id": chromosome.id,
            "generated_context": response.content
        }

    # 并行运行
    results = await asyncio.gather(*(generate_single(c) for c in chromosomes))
    
    # 我们存储部分结果以便稍后与分数结合
    eval_results = []
    for res in results:
        eval_results.append(EvaluationResult(
            chromosome_id=res["chromosome_id"],
            generated_context=res["generated_context"],
            fitness_score=-1.0,
            reasoning="等待评估"
        ))
        
    return {"evaluation_results": eval_results}

async def fitness_evaluation_node(state: EvolutionState) -> Dict[str, Any]:
    """评估生成的上下文 (Evaluates the generated contexts)."""
    # 评估模型可以配置为使用 decomposer_model 或专门的 evaluator_model
    evaluator_llm = LLMFactory.create(default_config.decomposer_model)
    evaluator_llm = evaluator_llm.with_structured_output(TargetFeature) # 复用 TargetFeature 结构来提取特征
    
    print(f"--- [节点] 适应度评估 (Fitness Evaluation) ---", flush=True)
    results = state["evaluation_results"]
    ideal_profile = set(state["ideal_context_profile"])
    
    async def evaluate_single(result: EvaluationResult):
        # 1. 反向画像：提取生成内容的实际特征
        chain = REVERSE_PROFILING_PROMPT | evaluator_llm
        
        try:
            output: TargetFeature = await chain.ainvoke({
                "target_description": state["target_description"],
                "generated_result": result.generated_context
            })
            actual_features = output.features
        except Exception as e:
            print(f"    [Warning] 特征提取失败: {e}", flush=True)
            actual_features = []

        # 2. 计算适应度 (Jaccard Similarity)
        # 简单的重合度计算：交集大小 / 理想特征集大小
        # 这里做一个简单的文本模糊匹配或完全匹配
        # 为了更鲁棒，这里假设 LLM 提取的特征是自然语言，可能无法精确匹配字符串。
        # 实际生产中可能需要用 Embedding 计算相似度。
        # 这里简化为：如果实际特征中有词语与理想特征重叠，就算匹配。
        
        matched_count = 0
        matches = []
        for ideal in ideal_profile:
            for actual in actual_features:
                # 简单的包含关系检查
                if ideal in actual or actual in ideal:
                    matched_count += 1
                    matches.append(f"{ideal}≈{actual}")
                    break
        
        # 避免除以零
        denominator = len(ideal_profile) if len(ideal_profile) > 0 else 1
        fitness_score = (matched_count / denominator) * 100.0
        
        reasoning = f"匹配特征: {matches}. 提取到的特征: {actual_features}"
        
        return EvaluationResult(
            chromosome_id=result.chromosome_id,
            generated_context=result.generated_context,
            actual_features=actual_features,
            fitness_score=fitness_score,
            reasoning=reasoning
        )

    scored_results = await asyncio.gather(*(evaluate_single(r) for r in results))
    
    # 寻找最佳结果
    best_score = -1.0
    best_context = state.get("best_context")
    
    for res in scored_results:
        if res.fitness_score > best_score:
            best_score = res.fitness_score
            best_context = res.generated_context
            
    print(f"    本轮最高分: {best_score:.2f}", flush=True)
            
    return {
        "evaluation_results": scored_results,
        "best_context": best_context
    }

async def evolution_analysis_node(state: EvolutionState) -> Dict[str, Any]:
    """更新基因库并检查收敛情况 (Updates gene pool and checks for convergence)."""
    print(f"--- [节点] 进化分析 (Evolution Analysis) ---", flush=True)
    
    updated_pool = update_gene_pool_stats(
        state["gene_pool"],
        state["evaluation_results"],
        state["chromosomes"],
        default_config.min_mutation_rate,
        default_config.max_mutation_rate
    )
    
    # 简单的分析打印
    scores = [r.fitness_score for r in state["evaluation_results"]]
    if scores:
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        print(f"    分析: 最高分 {max_score}, 平均分 {avg_score:.2f}", flush=True)
    
    return {"gene_pool": updated_pool}
