# main.py

from src.agent_v1 import graphv1
from src.agent_v2 import graphv2
import asyncio

if __name__ == "__main__":
    # response = asyncio.run(graphv1.ainvoke({"messages":[{"role":"user","content":"帮我搜索下LLM的记忆相关内容"}]}))
    response = asyncio.run(graphv2.ainvoke({"messages":[{"role":"user","content":"帮我搜索下LLM的记忆相关内容"}]}))
    print(response)
