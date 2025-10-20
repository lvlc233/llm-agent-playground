from typing import Annotated, Any, List,Optional,Callable
import asyncio
import threading
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage,BaseMessage,RemoveMessage
from langchain_core.runnables.graph import Node
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langchain.agents import ToolNode
from dotenv import load_dotenv
from langgraph.types import Command
from langchain_core.tools import tool

from enum import Enum
"""
初始化LLM
"""
# 加载环境变量
load_dotenv()
import os
model_name=os.getenv('OPENAI_MODEL_NAME')


"""
工具定义
"""
#工具
@tool("网络搜索")
async def web_search(query: str,tool_call_id:str)->Command:
    """用于搜索互联网"""
    await asyncio.sleep(3)
    res= f"网络搜索结果: {query}..."
    command= Command(
        update={"done_task": [ToolMessage(content=res,tool_call_id=tool_call_id)]},
    )
    return command
#工具
@tool("本地搜索")
async def local_search(query: str,tool_call_id:str)->Command:
    """用于搜索本地文件"""
    await asyncio.sleep(5)
    res= f"本地搜索结果: {query}..."
    command= Command(
        update={"done_task": [ToolMessage(content=res,tool_call_id=tool_call_id)]},
    )
    return command
tools=[web_search,local_search]
llm = ChatOpenAI(model=model_name, temperature=0.7).bind_tools(tools)
def call(state:MessagesState)->MessagesState:
    res=llm.invoke(state["messages"])
    state["messages"].append(res)
    return state 
graph=StateGraph(MessagesState)
graph.add_node("llm", call)
graph.add_edge(START, "llm")
graph.add_edge("llm", END)

app=graph.compile()
res=app.invoke({"messages": [HumanMessage(content="你好")]}) 
print(res["messages"]) 
