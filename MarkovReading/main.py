
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage,SystemMessage,HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph ,END
from typing import List,Annotated
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os


load_dotenv()
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv('LANGSMITH_API_KEY')
os.environ["LANGSMITH_PROJECT"] = os.getenv('LANGSMITH_PROJECT')

system_prompt = """
系统提示词:
    <指令:Agent>
        Agent阅读着标记为</可见>范围内容的文本
        Agent将根据</可见>的内容和旧印象生成印象,注意,Agent的记忆不会保留。
        </可见>的数据结构为
            index: 当前阅读的索引
            text: 你当前阅读的文本
        </印象>是你阅读以来的印象
    </指令:Agent>
    <指令:印象>
        Agent阅读</可见>的文本,Agent结合原本的</印象>,对</可见>的文本产生了一定的理解,并可以用简洁的话对其进行总结
        Agent尝试将这段印象与之前的印象进行整合。
    </指令:印象>
        <印象>
            {impression}
        </印象>
        </阅读>
            {read}
        <阅读>
    
"""


llm=init_chat_model(
        model="openai:"+os.getenv("OPENAI_MODEL_NAME"),
    ) 
class State(BaseModel):
    """状态类,用于存储状态"""
    messages: Annotated[List[AnyMessage], add_messages]
    chunk_index: int = 0
    chunk: List[str] = []
    chunk_max_size: int=0
    impression: str = ""


def chunk_cut(state: State):
    """将输入文本切分成多个块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    state.chunk=splitter.split_text(state.messages[-1].content)
    state.chunk_max_size=len(state.chunk)
    return state

def llm_call(state: State):
    """LLM 调用节点"""
    # 1,获取当前阅读的文本
    current_text=f"chunk_index:{state.chunk_index},text:往下读,你看到{state.chunk[state.chunk_index]}"
    # 2,获取上一轮的`印象`
    impression_text=state.impression
    # 2,调用LLM
    agent=llm
    response =agent.invoke([SystemMessage(content=system_prompt.format(
        impression=impression_text,
        read=current_text))])
    print("=====================")
    print(response.content)
    print("=====================")
    return {"messages": [response],
            "impression":response.content,
            "chunk_index":state.chunk_index+1
            }


def goto_next_or_end(state: State):
    if state.chunk_index<state.chunk_max_size-1:
        return "llm_call"
    return END

graph_build = StateGraph(State)
graph_build.add_node("chunk_cut",chunk_cut)
graph_build.add_node("llm_call",llm_call)
graph_build.add_edge(START,"chunk_cut")
graph_build.add_edge("chunk_cut","llm_call")
graph_build.add_conditional_edges(
    "llm_call",
    goto_next_or_end,
    {   
        "llm_call":"llm_call",
        END:END
    }
)



graph = graph_build.compile(name="agent")

info=graph.invoke({"messages":[HumanMessage(content="""

你阅读文章:
"
大型语言模型（LLMs）能否产生新颖的研究想法？
一项针对 100 多名自然语言处理（NLP）研究人员的大规模人类研究。Chenglei Si, Diyi Yang, Tatsunori Hashimoto，斯坦福大学 {clsi, diyiy, thashim}@stanford.edu。摘要：大型语言模型（LLMs）的最新进展引发了人们对其加速科学发现的潜力的乐观情绪，越来越多的工作提出了能够自主生成和验证新想法的研究智能体。
尽管如此，还没有任何评估表明 LLM 系统能够采取产生新颖的专家级想法的第一步，更不用说执行整个研究过程了。
我们通过建立一个实验设计来解决这个问题，该设计评估研究想法的产生，同时控制混杂因素，并对专家 NLP 研究人员和 LLM 构思智能体进行首次正面比较。
通过招募 100 多名 NLP 研究人员来撰写新颖的想法，并对 LLM 和人类的想法进行盲审，我们获得了关于当前 LLM 在研究构思方面的能力的第一个具有统计意义的结论：我们发现 LLM 产生的想法被认为比人类专家的想法更具新颖性（p < 0.05），而在可行性方面则略逊一筹。
通过仔细研究我们的智能体基线，我们发现了构建和评估研究智能体方面的开放性问题，包括 LLM 自我评估的失败以及它们在生成方面的缺乏多样性。
最后，我们承认人类对新颖性的判断可能很困难，即使是专家也是如此，并提出了一个端到端的研究设计，该设计招募研究人员将这些想法执行到完整的项目中，使我们能够研究这些新颖性和可行性判断是否会导致研究结果的显着差异。
1.1 介绍 LLM 的快速改进，尤其是在知识和推理等能力方面，使得其能够在科学任务中实现许多新的应用，例如解决具有挑战性的数学问题（Trinh et al., 2024），协助科学家撰写证明（Collins et al., 2024），检索相关工作（Ajith et al., 2024, Press et al., 2024），生成代码以解决分析或计算任务（Huang et al., 2024, Tian et al., 2024）以及发现大型文本语料库中的模式（Lam et al., 2024, Zhong et al., 2023）。
虽然这些是有用的应用，可以潜在地提高研究人员的生产力，但 LLM 是否能够承担研究过程中更具创造性和挑战性的部分仍然是一个悬而未决的问题。
我们专注于衡量 LLM 的研究构思能力的问题，并提出以下问题：当前的 LLM 是否能够产生与人类专家相当的新颖想法？
虽然构思只是研究过程的一部分，但这是一个需要回答的关键问题，因为它是科学研究过程的第一步，并且是对创建自己想法的自主研究智能体可能性的一个试金石。
评估 LLM 系统的专家级能力具有挑战性（Bakhtin。
我们发布了我们的智能体实现和所有人类审查分数，网址为：https://github.com/NoviScl/AI-Researcher。
*最后两位作者对本项目提供了同等程度的指导。
1 arXiv:2409.04109v1 [cs.CL]
2024 年 9 月 6 日 7 个 NLP 主题。
"

""")]})