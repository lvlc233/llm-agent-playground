from pydantic import BaseModel, Field
from typing import Optional, List
import uuid

class Gene(BaseModel):
    """
    Agent的基因代码
    逻辑：定义Agent的核心特征和行为模式
    目的：作为遗传算法的基本单位，包含身份、策略和记忆三个组成部分
    """
    identity: str = Field(..., description="Agent的角色身份和核心特性")
    strategy: str = Field(..., description="Agent的思考过程和方法论")
    memory: str = Field("", description="从前代继承的压缩知识")

    def to_prompt_string(self) -> str:
        """
        将基因转换为提示词字符串
        逻辑：格式化基因的三个组成部分
        目的：为LLM提供结构化的上下文信息
        """
        return f"""
[身份]
{self.identity}

[策略]
{self.strategy}

[祖先记忆]
{self.memory}
"""

class AgentConfig(BaseModel):
    """
    Agent实例的配置
    逻辑：包含Agent的运行时配置参数
    目的：管理Agent的生命周期和初始状态
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])  # 生成8位唯一ID
    gene: Gene                          # Agent的基因配置
    initial_energy: int = 100           # 初始能量值
    generation: int = 1                 # 当前代数

class EnvironmentStep(BaseModel):
    """
    环境序列中的单个步骤
    逻辑：定义环境步骤的数据结构
    目的：组织多步骤环境场景，支持顺序执行
    """
    step_id: str         # 步骤唯一标识符
    content: str        # 步骤内容文本（Agent可见）
    rubric: str = ""    # 裁判评分标准（Agent不可见，仅裁判可见）
    order: int          # 执行顺序
    file_path: str      # 文件路径
