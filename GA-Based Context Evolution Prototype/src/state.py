from typing import TypedDict, List, Optional, Annotated
from langgraph.graph.message import add_messages
from src.models import Gene, EnvironmentStep

class AgentState(TypedDict):
    """
    Agent状态数据结构
    逻辑：定义Agent在模拟过程中的所有状态信息
    目的：作为LangGraph状态机的核心数据结构，跟踪Agent的生存状态和交互历史
    """
    # Agent身份与状态
    agent_id: str              # Agent唯一标识符
    gene: Gene                 # Agent的基因配置
    generation: int            # 当前代数
    
    # 生存指标
    energy: int                # 当前能量值
    is_alive: bool             # 是否存活
    cause_of_death: Optional[str]  # 死亡原因（如果死亡）
    
    # 环境上下文
    current_step_index: int                          # 当前环境步骤索引
    environment_steps: List[EnvironmentStep]         # 完整的环境步骤序列
    current_environment_content: str                 # 当前环境内容
    
    # 交互历史
    messages: Annotated[List[dict], add_messages]   # LangGraph标准消息历史
    
    # 反馈/判断
    last_action_valid: bool      # 上次行动是否有效
    feedback: str                # 裁判反馈信息
    solved_steps: List[str]      # 已解决的步骤ID列表
