
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, AnyMessage,SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph ,END
from typing import List,Annotated,cast
from src.prompt import tool_kit_prompt
from src.config import get_model, get_think_model,logging,get_model_think_v3_5
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

# 提取 meta_thinks 的 keys 用于 Literal 类型
class State(BaseModel):
    """状态类,用于存储状态"""
    messages: Annotated[List[AnyMessage], add_messages] = []
    think_prompt: Annotated[str, Field(default="")] = ""
async def master(state:State)->State:
    logging.info(f"主节点准备")
    """
    主模型,用于对话的模型和进行决策的模型块
    """
    sys_message=f"""
        <思考>
            关于思考,
            你总是利用工具进行思考而非直接文本推理
            你总是习惯先去想法而非下定论
            每个思考对象总是明确的独立的,它总是最简的指向所思的那个对象的。
            思考的内容总是至少作为一条完整句子可理解的
            一个复杂的思考总是又若干简单的思考迭代而成的。
        </思考>
    """
    history=state.messages
    think_model=get_think_model(tool_kit_prompt)

    response =cast(AIMessage, await think_model.ainvoke([SystemMessage(content=sys_message),*history]))
    logging.info(f"主节点输出结果:content{response.content},tool_call{response.tool_calls}")
    return {"messages": [response]}

async def think(state:State)->State:
    logging.info(f"思考节点预备")
    """思考节点,用于思考"""
    think_prompt=state.think_prompt
    history=state.messages
    think_model=get_model_think_v3_5()
    base_think_prompt="""
        Agent不会的内容写长,因为如果写得长了,Agent会觉得没有价值;
        Agent用简洁清晰的方式表达当前的思考;
        每一段思考总是简单的;
        当前的输出是Agent的独白,只给Agent自己看，其他人看不到,
    """
    response =cast(AIMessage,await think_model.ainvoke([SystemMessage(content=base_think_prompt),SystemMessage(content=think_prompt)]))
    logging.info(f"思考节点输出结果:content{response.content}")
    return {"messages": [SystemMessage(f"Agent 想到:`{response.content}`")]}

async def tool_call(state:State):
    """工具节点,用于执行工具"""
    if (state.messages[-1].tool_calls):
        logging.info(f"继续思考")
        tool_node = ToolNode(tools=tool_kit_prompt)
        result = await tool_node.ainvoke(state)
        if (result):
            # 提取工具调用结果中的内容作为思考提示
            think_prompt_content = result["messages"][-1].content
            logging.info(f"思考方式{think_prompt_content}")
            return Command(
                    goto="think",
                    update={"think_prompt": think_prompt_content}
                )
    logging.info(f"结束思考")
    return Command(
                goto=END,
            )



graph_build = StateGraph(State)
graph_build.add_node("master",master)
graph_build.add_node("tool_call",tool_call)
graph_build.add_node("think",think) 
graph_build.add_edge(START,"master")
graph_build.add_edge("master","tool_call")
graph_build.add_edge("think","master")



graphv3_5 = graph_build.compile(name="agent").with_config(
            config={
                "recursion_limit": 100
            }
        )