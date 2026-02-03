
from pydantic import BaseModel, Field
from typing import Literal,Optional
from langchain_core.messages import AnyMessage,SystemMessage,AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph ,END
from typing import List,Annotated
from src.config import get_model,logging

# 思考的类型
meta_thinks={"association":"当你已经有一定数量的想法,并注意到他们之间存在某些关系时,你将尝试将这些想法联系起来,并将输出最有价值的最多为前3条关联及其关系",
    "deduction":"当你有了一个假设时,或来自你尝试的思考内容,尝试在已知的内容上进行逻辑或事实推理,并输出这些假设的推理过程和推理结果",
    "induction":"当你已经有一定的想法的时候,你可以分析这些想法之间是否存在某些相同的模式或结构",
    "analogy":"当你发现或认为某些概念和另外已经存在的概念具有某种结构上的相似性,你将尝试将概念结构的特性映射到新的概念上",
    "reflexivity":"当你发现可能存在某一个想法尝试错误的时候,你将尝试反思这个想法,并尝试找到错误的原因,并尝试修正这个想法"
    }
meta_thinks_cn={"association":"关联",
                "deduction":"演绎",
                "induction":"归纳",
                "analogy":"类比",
                "reflexivity":"反思"}
meta_think_types=list(meta_thinks.keys())
sys_think=SystemMessage(content="<system>Agent已学会自然语言的表达,但Agent对一些专业的知识不了解,这是一件好事,因为Agent可以学会自己探索知识,而不是依赖于固定的知识了,<system>")
# 提取 meta_thinks 的 keys 用于 Literal 类型
class State(BaseModel):
    """状态类,用于存储状态"""
    messages: Annotated[List[AnyMessage], add_messages] = []
    action: Optional[str] = None
    think_type: Optional[Literal[*meta_think_types]] = None
# 这是一个枚举问题,llm中像是让LLM固定输出一个范围内的固定的若干个选项,即选择的最优解是什么呢?
# 回答:就是结构化输出 

class Command_t(BaseModel):
    command: Literal["think", "speak"] = Field(description="""如果你需要思考,则输出`think` 
                                                            若你决定表达观点给用户,则输出`speak`,
                                                            """)

# 指令的目的是:识别thinks和speak,因此这是一个分类任务,分类任务的最佳做法是结构化输出,因此这里选择结构化输出
async def command_build(state: State):
    logging.info("指令识别中")
    history=state.messages
    llm=get_model().with_structured_output(Command_t)
    
    # 将历史消息整合为单个消息内容，避免JSON模式下的前缀错误
    history_content = "\n".join([f"{msg.type}: {msg.content}" for msg in history])
    prompt = f"{sys_think.content}\n\nConversation history:\n{history_content}\n\nBased on this, decide whether to think or speak."
    
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    state.action=response.command
    logging.info(f"指令识别结果: {response.command}")
    return state

# 具体的思考内容
class ThinkSelect(BaseModel):
    think_type: Literal[*meta_think_types] = Field(description=f"""现在,你看到了历史记录并想要进行思考,
                                                                 你将从以下的一种方式中选择一种最适合当前任务和你的性格的方式,
                                                                {meta_thinks}""")

async def think_select(state: State):
    """思考选择节点,用于选择思考类型"""
    logging.info("思考选择中")
    history=state.messages
    llm=get_model().with_structured_output(ThinkSelect)
    
    # 将历史消息整合为单个消息内容，避免JSON模式下的前缀错误
    history_content = "\n".join([f"{msg.type}: {msg.content}" for msg in history])
    prompt = f"Based on the conversation history:\n{history_content}\n\nPlease select an appropriate thinking type."
    
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    state.think_type=response.think_type
    logging.info(f"思考选择结果: {response.think_type}")
    return state

async def think_build(state: State):
    """思考节点,用于思考"""
    logging.info("思考中")
    print(meta_thinks[state.think_type])
    history=state.messages
    llm=get_model()
    response = await llm.ainvoke([*history,sys_think,SystemMessage(content=f"当前阶段你正在思考,而非和用户进行交流的角度出发。you see the history and  you want use the think type {state.think_type} think something,{meta_thinks[state.think_type]},你将以简洁的方式输出你的思考,思考的内容为Agent自己呈现,也就是Agent的自言自语,不能是长段落,因为长段落看上去不像是在思考,")])
    state.messages=[AIMessage(content="ai:"+response.content)]
    logging.info(f"思考结果: {response.content}")
    return state


async def speak_build(state: State):
    """说话节点,用于输出最终结果"""
    logging.info("说话中")
    history=state.messages
    llm=get_model()
    response = await llm.ainvoke([*history,sys_think,SystemMessage(content=f"you see the history and  you want speak something to user")])
    state.messages=[AIMessage(content="ai:"+response.content)]
    logging.info(f"说话结果: {response.content}")
    return state

graph_build = StateGraph(State)
graph_build.add_node("command",command_build)
graph_build.add_node("t_select",think_select)
graph_build.add_node("t_build",think_build) 
graph_build.add_node("s_build",speak_build)

graph_build.add_edge(START,"command")
graph_build.add_conditional_edges("command",
                                lambda state: "t_select" if state.action=="think" else "s_build",
                                {"t_select":"t_select","s_build":"s_build"}
                            )
graph_build.add_edge("t_select","t_build")
graph_build.add_edge("t_build","command")
graph_build.add_edge("s_build",END)



graphv2 = graph_build.compile(name="agent").with_config(
            config={
                "recursion_limit": 100
            }
        )
