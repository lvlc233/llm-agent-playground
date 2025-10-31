
from ast import Dict
from langchain_core.runnables.graph import Node
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, AnyMessage,SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph ,END
from typing import List,Annotated,cast,Set
from src.prompt import tool_kit_prompt
from src.config import get_think_model,logging
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

config={
    "think_batch_size": 5

}

class Think(BaseModel):
    """
    内容
    摘要
    权重
    """
    content:str
    summary:str=""
    w:float=0

class State(BaseModel):
    """状态类,用于存储状态"""
    messages: Annotated[List[AnyMessage], add_messages] = []
    memory:List[Think]|None=None
    think_batch:List[AIMessage]=[]

async def think(state:State)->State:
    cache=[]
    for _ in range(config["think_batch_size"]):
        cache.append([SystemMessage(content=f"{state.memory}"),*state.messages])
    think_model=get_think_model(tool_kit_prompt)
    think_batch = await think_model.abatch(cache)
    return {"think_batch":think_batch} 

async def tool_call(state:State):
    """工具节点,用于执行工具"""
    if (state.messages[-1].tool_calls):
        logging.info(f"继续思考")
        tool_node = ToolNode(tools=tool_kit_prompt)
        result = await tool_node.abatch(state)
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