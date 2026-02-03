"""
Tool definitions for thin_king project.

This module contains all the thinking tools used across different experiment versions.
"""

from typing import List
from langchain_core.tools import tool


# Core thinking tools
@tool
def metacognition(object_of_think: str) -> str:
    """
    对认知的认知，反思自己的思考模式
    
    Args:
        object_of_think: 进行metacognition的思考对象
    
    Returns:
        思考结果
    """
    return f"正在进行元认知思考: {object_of_think}"


@tool
def simple_think(object_of_think: str) -> str:
    """
    简单思考，复述或最基础的思考
    
    Args:
        object_of_think: 进行simple_think的思考对象
    
    Returns:
        思考结果
    """
    return f"正在进行简单思考: {object_of_think}"


@tool
def wait_i_think(object_of_think: str) -> str:
    """
    缓一下，觉得不对劲时暂停
    
    Args:
        object_of_think: 你觉得不对劲的点
    
    Returns:
        思考结果
    """
    return f"等等，让我想想: {object_of_think}"


@tool
def critical_think(object_of_think: str) -> str:
    """
    批判思考，质疑并深入分析
    
    Args:
        object_of_think: 进行批判思考的对象
    
    Returns:
        思考结果
    """
    return f"正在进行批判性思考: {object_of_think}"


@tool
def causal_think(object_of_think: str) -> str:
    """
    因果推理，梳理事物间的因果链
    
    Args:
        object_of_think: 你需要进行因果逻辑推理的思考对象
    
    Returns:
        思考结果
    """
    return f"正在进行因果推理: {object_of_think}"


@tool
def feeling() -> str:
    """
    放空一下，只是去感受的时候
    
    Returns:
        感受结果
    """
    return "正在感受当下..."


@tool
def silence() -> str:
    """
    沉默，不说任何话
    
    Returns:
        沉默状态
    """
    return "..."


@tool
def assume(assume_context: str) -> str:
    """
    假设某个内容时，思考一些没有考虑过的内容
    
    Args:
        assume_context: 你假设的内容
    
    Returns:
        思考结果
    """
    return f"假设: {assume_context}，让我思考一些新的可能性..."


@tool
def suspect(object_of_think: str) -> str:
    """
    怀疑某个内容时
    
    Args:
        object_of_think: 你怀疑的内容
    
    Returns:
        思考结果
    """
    return f"我对此表示怀疑: {object_of_think}"


# Meta-thinking tools (for v2 and later)
@tool
def association(object_of_think: str) -> str:
    """
    关联思考 - 将想法联系起来
    
    Args:
        object_of_think: 需要关联的思考对象
    
    Returns:
        关联结果
    """
    return f"正在建立关联: {object_of_think}"


@tool
def deduction(object_of_think: str) -> str:
    """
    演绎推理 - 从假设进行逻辑推理
    
    Args:
        object_of_think: 需要演绎的思考对象
    
    Returns:
        演绎结果
    """
    return f"正在进行演绎推理: {object_of_think}"


@tool
def analogy(object_of_think: str) -> str:
    """
    类比思考 - 寻找结构相似性
    
    Args:
        object_of_think: 需要类比的思考对象
    
    Returns:
        类比结果
    """
    return f"正在进行类比思考: {object_of_think}"


@tool
def induction(object_of_think: str) -> str:
    """
    归纳思考 - 寻找模式和结构
    
    Args:
        object_of_think: 需要归纳的思考对象
    
    Returns:
        归纳结果
    """
    return f"正在进行归纳思考: {object_of_think}"


@tool
def reflexivity(object_of_think: str) -> str:
    """
    反思思考 - 反思错误并修正
    
    Args:
        object_of_think: 需要反思的思考对象
    
    Returns:
        反思结果
    """
    return f"正在进行反思: {object_of_think}"


# Tool collections for different versions
basic_thinking_tools = [
    metacognition,
    simple_think,
    wait_i_think,
    critical_think,
    causal_think,
    feeling,
    silence,
    assume,
    suspect,
]

meta_thinking_tools = [
    association,
    deduction,
    analogy,
    induction,
    reflexivity,
]

# Combined tool kit
all_thinking_tools = basic_thinking_tools + meta_thinking_tools

# For backward compatibility
tool_kit = basic_thinking_tools
tool_kit_prompt = meta_thinking_tools
tool_kit_llm = all_thinking_tools