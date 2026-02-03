# main.py

import asyncio
from experiments import graphv1, graphv2, graphv3, graphv4, graphv5

async def test_version(graph, version_name, query):
    """测试指定版本的图"""
    print(f"\n=== 测试 {version_name} ===")
    try:
        response = await graph.ainvoke({"messages": [{"role": "user", "content": query}]})
        print(f"响应消息数量: {len(response['messages'])}")
        for i, message in enumerate(response["messages"]):
            print(f"消息 {i+1}: {message}")
    except Exception as e:
        print(f"错误: {e}")

async def main():
    """主函数，测试所有版本"""
    query = "如果让你设计Agent的记忆模块你会怎么设计?"
    
    # 测试所有版本
    await test_version(graphv1, "Version 1 (Basic ReAct)", query)
    await test_version(graphv2, "Version 2 (Meta-Thinking)", query)
    await test_version(graphv3, "Version 3 (Tool-Based)", query)
    await test_version(graphv4, "Version 4 (Memory-Enhanced)", query)
    await test_version(graphv5, "Version 5 (Network-Based)", query)

if __name__ == "__main__":
    asyncio.run(main())
