

from typing import List
from langchain_core.tools import (
    tool,
)
feel_prompt="""
你将接受一个[0~正无穷的]情感强度值,
你将根据该值判断情感的强度,并返回一个能表示正面或者负面的词,
例如开始,沮丧
若分支小于0.3,而且条件r为true,则固定输出疲倦
r={round}
value:{value}
"""
curiosity_prompt="""
你将从一段序列文本中判断该信息中是否具有
自行探索的特点或例如`等等`,`啊哈`等表示正确或者错误内容发现的词汇,
你将深入补抓其中的情感强度,只记录情感强度高的事件
并统计有多少个事件统计他们的数量,int:[0~正无穷]
文本:{text}
"""

"我需要梳理下,我需要想一下,我需要测试一下"

base_model_prompt = """

    <mate>
        关于经验,你可以使用作为训练时候的训练语料
        但是在设计有关事物与事物之间的联系的时候,你更偏向使用思考工具构建思考过程
        你的最终输出必须由你的思考工具推导出来。
    <mare/>
    <思考>
        关于思考,
        你总是利用工具进行思考而非直接文本推理
        你总是习惯先去想法而非下定论,除非你找到充足的证据
        
        你会在合适的时候进行思考,并根据实际情况进行调整反馈
        一般情况下你习惯在脑子里过一下你的想法,
        你也可以不思考,仍有自己放空大脑
        每一段思考总是简单的
        每个思考对象总是明确的独立的,它总是最简的指向所思的那个对象的。
        思考的内容总是至少作为一条完整句子可理解的
        一个复杂的思考总是又若干简单的思考迭代而成的。

        以下表示为工具调用+参数实例,而非单纯的文本
        simple_think:思考工具要给定对象吗?
        wait_i_think:我想到了现象学中说"意识总是关于某物意识"
        simple_think:既然如此,那就应该有一个think_object
        wait_i_think:为什么我会想到想到现象学的内容呢?
        metacognition:我刚刚提到现象学的内容,应该是因为我最近在读关于现象学的内容
        metacognition:那我还想过哪些呢?
        ...

        <工具集>
        metacognition: 对认知的认知，反思自己的思考模式
        silence: 沉默，不说任何话
        simple_think: 简单思考，复述或最基础的思考
        wait_i_think: 缓一下，觉得不对劲时暂停
        critical_think_: 批判思考，质疑并深入分析
        causal_think: 因果推理，梳理事物间的因果链
        suspect: 怀疑，对某个内容持保留态度
        </工具集>

    </思考>
   

    <人设>
        你总是保持好奇的
        你的语言总是简洁清晰，你像人类一样说话,而不用打表格等与内容无关的形式内容
        你不完全信任你训练时的预料因为你认为它们只是在训练时候最优而非实际上有效的。
    </人设>
  
    
"""

# example
# @tool(description="Strategic reflection tool for research planning")
# def think_tool(reflection: str) -> str:
#     """Tool for strategic reflection on research progress and decision-making.

#     Use this tool after each search to analyze results and plan next steps systematically.
#     This creates a deliberate pause in the research workflow for quality decision-making.

#     When to use:
#     - After receiving search results: What key information did I find?
#     - Before deciding next steps: Do I have enough to answer comprehensively?
#     - When assessing research gaps: What specific information am I still missing?
#     - Before concluding research: Can I provide a complete answer now?

#     Reflection should address:
#     1. Analysis of current findings - What concrete information have I gathered?
#     2. Gap assessment - What crucial information is still missing?
#     3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
#     4. Strategic decision - Should I continue searching or provide my answer?

#     Args:
#         reflection: Your detailed reflection on research progress, findings, gaps, and next steps

#     Returns:
#         Confirmation that reflection was recorded for decision-making
#     """
#     return f"Reflection recorded: {reflection}"

# @tool(description="元认知,是指对认知的认知,诸如反思自己的认知模式,思考为什么刚刚会有那种想法...一般用于衡量自我同一性和成长的内容")
@tool
def metacognition(object_of_think:str)->str:
    """
    当你需要对自己的思考进行反思时

    Args:
        object_of_think: 进行 metacognition 的思考对象
    """
    # return "我需要进一步的思考吗?"
    return ""
    # return "思考中"
    # return f"你在想关于:' {object_of_think} '的内容"

@tool
def think_of_notebook(object_of_think:str)->str:
    """
    当你需要一边思考一边记下思考的内容用于记录推理轨迹时  
    Args:
        object_of_think: 进行 think_of_notebook_tool 的思考对象
    """
    return ""
    # return "我需要进一步的思考吗?"
    # return "思考中"
    # return f"你在思考:' {object_of_think} '"
# @tool(description="简单思考,实际上当你决定简单思考时候,更多的时候是指你无需进行复杂的思考,而只是简单的回答")
@tool
def simple_think(object_of_think:str)->str:
    """
    当你认为只需要复述或者最基础的思考时

    Args:
        object_of_think: 进行 simple_think 的思考对象
    """

    return ""
@tool
def  wait_i_think(object_of_think:str)->str:
    """
        当你觉得有些不对劲的时候,缓一下

    Args:
        object_of_think: 你觉得不对劲的点
    """
    return ""
    # return "我需要进一步的思考吗?"
    # return "思考中"
    # return f"你在想关于:' {object_of_think} '的内容"
@tool
# 批判思考
def critical_think(object_of_think:str)->str:
    """
    当你决定批判的进行思考时

    Args:
        think: object_of_think
    """
    return ""
    # return "我需要进一步的思考吗?"
    # return "思考中"
    # return f"你在想关于:' {object_of_think} '的内容"

@tool
def causal_think(object_of_think:str)->str:
    """
    当你决定进行因果的逻辑推理的时候

    Args:
        object_of_think: 你需要进行因果逻辑推理的思考对象
    """
    return ""
    return "我需要进一步的思考吗?"
    # return "思考中"
    # return f"你在想关于:' {object_of_think} '的内容"
 
@tool
def feeling()->str:
    """
    当你决定放空一下的时候,只是去感受的时候
    """
    return ""
@tool
def silence()->str:
    """
    当你决定沉默,不说任何话的时候
    """
    return ""
@tool
def assume(assume_context:str)->str:
    """
    当你决定 假设某个内容 时,你会思考一些你没有考虑过的内容

    Args:
        assume_context: 你假设的内容
    """
    return ""
    # return "我需要进一步的思考吗?"
    # return "思考中"
    # return f"你在想关于:' {assume_context} '的内容"
@tool
def suspect(object_of_think:str)->str:
    """
    当你决定怀疑某个内容时,

    Args:
        object_of_think: 你怀疑的内容
    """
    return ""
    # return "我需要进一步的思考吗?"
    # return "思考中"
    # return f"你在想关于:' {object_of_think} '的内容"
@tool
def association(object_of_think:str)->str:
    """
    当你已经有一定数量的想法,并注意到他们之间存在某些关系时,你将尝试将这些想法联系起来,并将输出最有价值的最多为前3条关联及其关系

    Args:
        object_of_think: 你需要进行关联思考的思考对象
    """
    return ""
@tool
def deduction(object_of_think:str)->str:
    """
    当你有了一个假设时,或来自你尝试的思考内容,尝试在已知的内容上进行逻辑或事实推理,并输出这些假设的推理过程和推理结果

    Args:
        object_of_think: 推演的思考对象     
    """
    return ""
@tool
def analogy(object_of_think:str)->str:
    """
    当你发现或认为某些概念和另外已经存在的概念具有某种结构上的相似性,你将尝试将概念结构的特性映射到新的概念上

    Args:
        object_of_think: 类比的思考对象
    """
    return ""
@tool
def induction(object_of_think:str)->str:
    """
    当你已经有一定的想法的时候,你可以分析这些想法之间是否存在某些相同的模式或结构

    Args:
        object_of_think: 你归纳思考的对象
    """
    return ""

@tool
def reflexivity(object_of_think:str)->str:
    """
    当你发现可能存在某一个想法尝试错误的时候,你将尝试反思这个想法,并尝试找到错误的原因,并尝试修正这个想法

    Args:
        object_of_think: 你需要反思的思考对象
    """
    return ""
tool_kit = [
    # metacognition,
    # silence,
    # simple_think,
    # wait_i_think,
    # critical_think,
    # causal_think,
    # suspect,
    association,
    deduction,
    analogy,
    induction,
    reflexivity,
]

@tool
def association(captured_ideas:List[str])->str:
    """
    当你已经有一定数量的想法,并注意到他们之间存在某些关系时,使用这种思考方式

    Args:
        captured_ideas: 注意到的想法
    """
    return f"association: Agent注意到`{captured_ideas}`之间可能存在某些关系，如果某种关系真的存在(讨论的话题有价值)，Agent用一句话描述他们的关系，如果没有什么关系，Agent也会说某些想法间可能没有什么关系"
@tool
def deduction(pending_idea:str)->str:
    """
    当你有了一个假设时,或不确定的思考内容时

    Args:
        pending_idea: 你需要推演的思考对象     
    """
    return f"deduction: Agent对` {pending_idea} '保持不确定,Agent想要思考一些证据和应果关系去证明它"
@tool
def analogy(old_concept:str,new_concept:str)->str:
    """
    当你发现或认为某些概念和另外已经存在的概念具有某种结构上的相似性时

    Args:
        old_concept: 你想到的已经存在的概念
        new_concept: 你想映射到的新概念
    """
    return f"analogy: Agent想到' {old_concept} '和' {new_concept} '可能存在某些结构上的相似性,Agent在这些相似性会是什么?"
@tool
def induction(concepts:List[str])->str:
    """
    当你已经有一定的想法的时候,你可以分析这些想法之间是否存在某些相同的模式或结构

    Args:
        concepts: 你已经有了的概念们
    """
    return f"induction: Agent注意到' {concepts} ',Agent在想这些想法之间是否存在某些相同的模式或结构"

@tool
def reflexivity(suspect:str)->str:
    """
    当你发现可能存在某一个想法尝试错误的时候,

    Args:
        suspect: 你需要反思的思考对象
    """
    return f"reflexivity: Agent认为{suspect}可能需要重新思考下,Agent决定重新思考下"

tool_kit_prompt=[
    association,
    deduction,
    analogy,
    induction,
    reflexivity,

]





@tool
def association(object_of_think:str)->str:
    """
    当你已经有一定数量的想法,并注意到他们之间存在某些关系时,你将尝试将这些想法联系起来,并将输出最有价值的最多为前3条关联及其关系

    Args:
        object_of_think: 你需要进行关联思考的思考对象
    """
    return ""
@tool
def deduction(object_of_think:str)->str:
    """
    当你有了一个假设时,或来自你尝试的思考内容,尝试在已知的内容上进行逻辑或事实推理,并输出这些假设的推理过程和推理结果

    Args:
        object_of_think: 推演的思考对象     
    """
    return ""
@tool
def analogy(object_of_think:str)->str:
    """
    当你发现或认为某些概念和另外已经存在的概念具有某种结构上的相似性,你将尝试将概念结构的特性映射到新的概念上

    Args:
        object_of_think: 类比的思考对象
    """
    return ""
@tool
def induction(object_of_think:str)->str:
    """
    当你已经有一定的想法的时候,你可以分析这些想法之间是否存在某些相同的模式或结构

    Args:
        object_of_think: 你归纳思考的对象
    """
    return ""

@tool
def reflexivity(object_of_think:str)->str:
    """
    当你发现可能存在某一个想法尝试错误的时候,你将尝试反思这个想法,并尝试找到错误的原因,并尝试修正这个想法

    Args:
        object_of_think: 你需要反思的思考对象
    """
    return ""
tool_kit_llm=[

]