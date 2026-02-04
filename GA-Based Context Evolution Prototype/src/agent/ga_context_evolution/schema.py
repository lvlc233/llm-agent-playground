from typing import List, TypedDict, Optional, Dict
from pydantic import BaseModel, Field

class TargetFeature(BaseModel):
    """目标特征"""
    features: List[str] = Field(description="目标特征的具体描述列表（例如：'包含代码示例'，而不是'代码'）")

class ContextChunkList(BaseModel):
    """上下文块列表"""
    chunks: Dict[str, List[str]] = Field(description="上下文块列表,key是特征,value是该特征对应的上下文列表")

class ContextChunk(BaseModel):
    """遗传信息的最小单元（基因）。"""
    id: str = Field(description="该片段的唯一标识符")
    content: str = Field(description="该片段的实际文本内容")
    mutation_rate: float = Field(default=0.1, description="该片段被突变/交换的概率")
    usage_count: int = Field(default=0, description="该片段在通过高适应度筛选的上下文中被使用的次数")
    fitness_contribution: float = Field(default=0.0, description="累积的适应度贡献分数")

class Chromosome(BaseModel):
    """代表潜在上下文的基因序列（个体）。"""
    id: str = Field(description="该染色体的唯一标识符")
    chunk_ids: List[str] = Field(description="有序的片段ID列表")

class EvaluationResult(BaseModel):
    """染色体的评估结果（表型评估）。"""
    chromosome_id: str
    generated_context: str
    actual_features: List[str] = Field(default_factory=list, description="从生成结果中反向提取的实际特征")
    fitness_score: float = Field(description="0到100之间的分数")
    reasoning: str = Field(description="评分背后的推理")

class EvolutionState(TypedDict):
    """进化过程的全局状态。"""
    # 输入
    target_description: str
    
    # 元数据
    # 进化目标:由模型分解出来(或人工输入)
    ideal_context_profile: List[str]
    # 生成轮次 
    generation_round: int
    # 最大轮次
    max_rounds: int
    
    # 基因池（所有可用片段）
    gene_pool: Dict[str, List[ContextChunk]]
    
    # 种群（当前一代染色体）
    chromosomes: List[Chromosome]
    
    # 表型与评估
    evaluation_results: List[EvaluationResult]
    
    # 最终输出
    best_context: Optional[str]
    # 对应的序列
    best_chromosome: Optional[List[ContextChunk]]
