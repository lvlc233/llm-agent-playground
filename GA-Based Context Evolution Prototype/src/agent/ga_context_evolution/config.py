from dataclasses import dataclass

@dataclass
class EvolutionConfig:
    """进化过程和模型选择的配置。"""
    
    # 模型选择
    # 分解器
    decomposer_model: str = "Pro/deepseek-ai/DeepSeek-V3.2"  # 为了测试速度而更改
    # 生成器
    generator_model: str = "Pro/deepseek-ai/DeepSeek-V3.2"  
    # 评估器
    evaluator_model: str = "Pro/deepseek-ai/DeepSeek-V3.2"   # 为了测试速度而更改
    
    # 进化参数
    # 初始化数量
    initial_population_size: int = 5
    # 最大迭代次数
    max_generations: int = 5
    # 突变概率
    min_mutation_rate: float = 0.05
    max_mutation_rate: float = 0.8
    # 迭代结束条件
    convergence_threshold: float = 95.0  # 如果分数 > 95 则停止
    
    # 分块
    num_initial_chunks: int = 3

# 全局配置实例
default_config = EvolutionConfig()
