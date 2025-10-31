# main.py

from langchain_core import messages
from src.agent_v1 import graphv1
from src.agent_v2 import graphv2
from src.agent_v3 import graphv3
import asyncio

if __name__ == "__main__":
    # response = asyncio.run(graphv1.ainvoke({"messages":[{"role":"user","content":"帮我搜索下LLM的记忆相关内容"}]}))
    # response = asyncio.run(graphv2.ainvoke({"messages":[{"role":"user","content":"帮我搜索下LLM的记忆相关内容"}]}))
    response = asyncio.run(graphv3.ainvoke({"messages":[{"role":"user","content":"如果让你设计Agent的记忆模块你会怎么设计?"}]}))
    print( message+"\n" for message in response["messages"])
