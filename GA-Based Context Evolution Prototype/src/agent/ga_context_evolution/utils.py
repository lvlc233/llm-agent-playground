import random
import uuid
import json
import re
from typing import List, Dict, Any, Optional
from .schema import ContextChunk, Chromosome, EvaluationResult

def parse_json_output(text: str) -> Any:
    """帮助函数：解析LLM输出的JSON，处理可能的markdown代码块。"""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 简单回退：提取列表
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        # 回退：提取字典
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        return {}

def create_initial_chunks(content_list: List[str]) -> List[ContextChunk]:
    """将字符串列表转换为 ContextChunk 对象。"""
    chunks = []
    for content in content_list:
        chunks.append(ContextChunk(
            id=str(uuid.uuid4())[:8],
            content=content,
            mutation_rate=0.5, # 初始中性概率
            usage_count=0,
            fitness_contribution=0.0
        ))
    return chunks

def select_and_crossover(
    gene_pool: Dict[str, List[ContextChunk]], 
    features: List[str],
    population_size: int
) -> List[Chromosome]:
    """
    基于插槽（特征）的交叉逻辑。
    
    对于每一个新的染色体（个体）：
    1. 遍历每一个特征（插槽）。
    2. 从该特征对应的基因池中，根据突变率/适应度选择一个片段。
       - 好的片段（低突变率）有更高概率被选中。
    3. 组合这些片段形成一个完整的染色体。
    """
    chromosomes = []
    
    for _ in range(population_size):
        selected_ids = []
        
        for feature in features:
            chunks = gene_pool.get(feature, [])
            if not chunks:
                continue
                
            # 计算权重: 1 - mutation_rate
            # 确保最小权重
            weights = [max(1.0 - c.mutation_rate, 0.05) for c in chunks]
            
            # 从该特征的池中选择一个
            chosen_chunk = random.choices(chunks, weights=weights, k=1)[0]
            selected_ids.append(chosen_chunk.id)
            
        chromosomes.append(Chromosome(
            id=str(uuid.uuid4())[:8],
            chunk_ids=selected_ids
        ))
        
    return chromosomes

def update_gene_pool_stats(
    gene_pool: Dict[str, List[ContextChunk]], 
    results: List[EvaluationResult], 
    chromosomes: List[Chromosome],
    min_rate: float,
    max_rate: float
) -> Dict[str, List[ContextChunk]]:
    """
    根据评估结果更新突变率。
    现在 gene_pool 是一个字典，我们需要遍历字典中的所有列表。
    """
    
    # 建立 chunk_id -> fitness scores 的映射
    # 因为 gene_pool 是嵌套的，我们先扁平化处理映射关系，最后再写回
    
    chunk_fitness_map: Dict[str, List[float]] = {}
    
    # 初始化映射
    for chunks in gene_pool.values():
        for chunk in chunks:
            chunk_fitness_map[chunk.id] = []
            
    chromosome_map = {c.id: c for c in chromosomes}
    
    # 收集分数
    for res in results:
        chrom = chromosome_map.get(res.chromosome_id)
        if not chrom:
            continue
        for chunk_id in chrom.chunk_ids:
            if chunk_id in chunk_fitness_map:
                chunk_fitness_map[chunk_id].append(res.fitness_score)
    
    # 更新片段
    updated_pool = {}
    
    for feature, chunks in gene_pool.items():
        updated_chunks = []
        for chunk in chunks:
            scores = chunk_fitness_map.get(chunk.id, [])
            if scores:
                avg_score = sum(scores) / len(scores)
                chunk.usage_count += len(scores)
                chunk.fitness_contribution = avg_score
                
                # 更新突变率
                # 分数越高 -> 突变率越低 (保留)
                raw_rate = 1.0 - (avg_score / 100.0)
                chunk.mutation_rate = max(min_rate, min(max_rate, raw_rate))
            else:
                # 未使用的片段，保持原样或微调
                pass
            
            updated_chunks.append(chunk)
        updated_pool[feature] = updated_chunks
        
    return updated_pool
