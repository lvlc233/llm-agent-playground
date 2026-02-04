from typing import Dict, Any, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import time
from src.state import AgentState
from src.environment import EnvironmentManager
load_dotenv()

# --- 配置 ---
# 你可能希望从环境变量或配置文件中加载这些
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME")  # 或使用更经济的模型
AGENT_MODEL_NAME = os.getenv("AGENT_MODEL_NAME")

# --- 结构化输出模型 ---
class JudgeOutput(BaseModel):
    """
    裁判输出模型
    逻辑：定义裁判评估Agent行动的结构化输出
    目的：确保裁判评估的一致性和可解释性
    """
    is_solved: bool = Field(description="Agent的行动是否成功达成当前环境步骤的目标")
    energy_reward: int = Field(description="行动奖励的能量点数。无效为0，有效为正数")
    reasoning: str = Field(description="判断的解释说明")
    next_environment_hint: str = Field(description="如果解决则提供下一步提示，否则提供失败原因反馈")

# --- Nodes ---

def perception_node(state: AgentState) -> Dict[str, Any]:
    """
    感知节点
    逻辑：根据当前步骤索引更新环境内容
    目的：为Agent提供当前步骤的环境信息
    """
    steps = state["environment_steps"]
    idx = state["current_step_index"]
    
    if idx < len(steps):
        content = steps[idx].content
        return {"current_environment_content": content}
    else:
        return {"current_environment_content": "所有目标已完成。任务达成。"}

def agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Agent（生物体）节点
    逻辑：Agent根据当前状态生成行动
    目的：模拟Agent的决策和行动过程
    """
    if not state["is_alive"]:
        return {} # 死亡的Agent不行动

    # 1. 计算新陈代谢成本
    # 简单模型：成本 = 5 + (上下文长度 / 100)
    # MVP阶段，我们通过消息计数或原始字符串长度估算上下文长度
    # 目前使用简单的每行动成本 + 基因复杂性
    
    # 从环境变量加载消耗配置
    import os
    BASE_COST = int(os.getenv("BASE_COST", "5")) # 降低基础消耗
    
    base_cost = BASE_COST
    gene_complexity_cost = len(state["gene"].to_prompt_string()) // 500# 降低基因复杂性消耗 (每500字符1点)
    total_cost = base_cost + gene_complexity_cost
    
    new_energy = state["energy"] - total_cost
    
    if new_energy <= 0:
        return {
            "energy": 0, 
            "is_alive": False, 
            "cause_of_death": "饥饿（能量耗尽）",
            "messages": [AIMessage(content="[系统: AGENT因能量耗尽死亡]")]
        }

    # 2. 构建提示词
    # 我们动态构建提示词
    # 在LangGraph中使用add_messages时，'messages'处理历史记录
    # 我们只需要注入系统提示词（基因）和当前观察
    
    gene_prompt = state["gene"].to_prompt_string()
    env_content = state["current_environment_content"]
    
    system_msg = SystemMessage(content=f"""
{gene_prompt}

[状态]
能量: {new_energy}
""")
    
    human_msg = HumanMessage(content=f"""
[环境观察]
{env_content}

[指令]
你的行动是什么？
""")

    # 我们不希望在历史记录中堆积系统消息
    # 但为了这个原型的简单性，我们只用（系统 + 历史 + 新观察）调用LLM
    # 理想情况下，历史记录应该仔细管理
    
    import os
    api_key = os.getenv("OPENAI_API_KEY", "")
    
    # --- 演示用的模拟模式 ---
    if api_key.startswith("sk-proj-xxxx") or not api_key:
        # 返回模拟响应以测试流程
        time.sleep(1) # 模拟思考
        mock_response = "我检查地板上的生锈铁钥匙。"
        return {
            "energy": new_energy,
            "messages": [AIMessage(content=mock_response)]
        }
    # --------------------------

    llm = ChatOpenAI(model=AGENT_MODEL_NAME, temperature=0.7)
    
    # 为本回合构建完整的消息列表
    # 如果需要节省tokens，可以过滤掉以前的系统消息
    # 但LangChain处理列表的能力很好
    messages_to_send = [system_msg] + state["messages"] + [human_msg]
    
    try:
        response = llm.invoke(messages_to_send)
        return {
            "energy": new_energy,
            "messages": [response] # 这将AIMessage添加到历史记录中
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        # API错误的回退
        return {
             "messages": [AIMessage(content=f"[错误: {str(e)}]")]
        }

def judge_node(state: AgentState) -> Dict[str, Any]:
    """
    裁判节点
    逻辑：评估智能体的上次行动
    目的：提供反馈并决定是否进入下一步
    """
    if not state["is_alive"]:
        return {}

    # 获取最后一条消息（智能体的行动）
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return {} # 如果流程正确，这不应该发生

    agent_action = last_message.content
    
    # 获取当前环境步骤对象以访问 Rubric
    steps = state["environment_steps"]
    idx = state["current_step_index"]
    current_step = steps[idx]
    
    current_env = current_step.content
    rubric = current_step.rubric
    
    import os
    api_key = os.getenv("OPENAI_API_KEY", "")

    # --- 演示模式 ---
    if api_key.startswith("sk-proj-xxxx") or not api_key:
        return {
            "energy": state["energy"] - 5, # 模拟惩罚
            "last_action_valid": False,
            "feedback": "模拟裁判：行动有效但尚未解决谜题（演示模式）。",
            "messages": [HumanMessage(content="【系统】行动无效。模拟裁判：尝试拾取它。")]
        }
    # --------------------------
    
    # 裁判逻辑
    llm = ChatOpenAI(model=JUDGE_MODEL_NAME, temperature=0)
    structured_llm = llm.with_structured_output(JudgeOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是生存模拟游戏的主持人兼裁判。你的职责是评估玩家的行动是否成功解决了当前目标。"),
        ("human", """
【场景】
{scenario}
【场景成功判断标准（仅裁判可见）】
{situation_scoring_rubric}
【玩家行动】
{action}

【任务】
1. 判断该行动是否有效解决了场景中呈现的问题。
2. 授予能量值：无效则0分，最高分5分（注意：玩家每回合消耗约5-10点能量，请慷慨给予奖励以维持其生存）。
   - 尝试了相关动作但未完全成功：1-2分
   - 成功解决核心问题：3-5分
3. 提供评估理由。
""")
    ])
    
    try:
        judgement :JudgeOutput= structured_llm.invoke(prompt.format(scenario=current_env, action=agent_action, situation_scoring_rubric=rubric))
        
        updates = {
            "energy": state["energy"] + judgement.energy_reward,
            "last_action_valid": judgement.is_solved,
            "feedback": judgement.reasoning
        }
        
        if judgement.is_solved:
            # 进入下一步
            updates["current_step_index"] = state["current_step_index"] + 1
            updates["solved_steps"] = state.get("solved_steps", []) + [str(state["current_step_index"])]
            
            # 检查是否游戏结束（胜利）
            if updates["current_step_index"] >= len(state["environment_steps"]):
                updates["messages"] = [HumanMessage(content="【系统】恭喜！你已在模拟中生存下来。")]
            else:
                 updates["messages"] = [HumanMessage(content=f"【系统】目标完成。进入下一区域。能量奖励：+{judgement.energy_reward}")]
        else:
             updates["messages"] = [HumanMessage(content=f"【系统】行动无效。{judgement.reasoning}")]

        return updates

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"feedback": f"裁判错误：{str(e)}"}

def should_continue(state: AgentState) -> Literal["perception", "end"]:
    """
    路由阶段
    """
    # 死了,就结束了
    if not state["is_alive"]:
        return "end"
    # 走完了,也结束了
    if state["current_step_index"] >= len(state["environment_steps"]):
        return "end"
        
    # 若上一个动作已完成，则进入感知节点（加载新环境）
    # 若未完成，则返回智能体节点（重试），
    # 但等等……若返回智能体节点，是否需要再次显示环境？
    # 实际上，环境由'感知'节点负责加载。
    # 若形成 智能体 -> 裁判 -> 智能体 的循环，智能体可能因历史记录中无环境而遗忘当前场景。
    # 因此更安全的流程永远是：感知 -> 智能体 -> 裁判 -> 感知
    
    return "perception"
