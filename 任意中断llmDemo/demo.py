from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import asyncio

# 加载.env文件
load_dotenv(dotenv_path=".env",override=True)

# # # 确保 LangSmith 追踪启用--->只有这么做了才会开始追踪
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv('LANGSMITH_ENDPOINT')
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGSMITH_API_KEY')
os.environ["LANGCHAIN_PROJECT"] = os.getenv('LANGSMITH_PROJECT')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_BASE_URL"] = os.getenv('OPENAI_BASE_URL')
os.environ["OPENAI_MODEL_NAME"] = os.getenv('OPENAI_MODEL_NAME')

llm = ChatOpenAI(model_name=os.getenv('OPENAI_MODEL_NAME'))

class State(TypedDict):
    messages: dict[str,str]

def nodeA(state:State,writer:BaseMessage):
    state["messages"]["user"]="你好"
    return state

def nodeB(state:State):
    state["messages"]["assistant"]="你好"
    return state
graph=StateGraph(State)
graph.add_node("A",nodeA)
graph.add_node("B",nodeB)
graph.add_edge(START,"B")
graph.add_edge(START,"A")
