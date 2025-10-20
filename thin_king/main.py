# main.py

from src.agent import graph
import asyncio

if __name__ == "__main__":
    response = asyncio.run(graph.ainvoke({"messages":[{"role":"user","content":"帮我搜索下LLM的记忆相关内容"}]}))
    print(response)
