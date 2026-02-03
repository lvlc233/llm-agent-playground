import asyncio
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from src.agent.ga_context_evolution.agent import graph
from src.utils.monitor import StateMonitor

async def main():
    load_dotenv()
    
    target = "请写一段关于‘Rust语言内存安全机制’的简短介绍，要求通俗易懂，适合初学者，且包含‘所有权’和‘借用’两个概念。"
    
    print(f"目标: {target}\n")
    print("开始进化过程...\n")
    
    initial_state = {
        "target_description": target
    }
    
    final_state = initial_state.copy()
    
    try:
        # 使用 astream 实时获取每个节点的更新
        async for event in graph.astream(initial_state):
            for node_name, updates in event.items():
                # 打印结构化日志
                StateMonitor.print_step(node_name, updates)
                
                # 更新本地状态以便最后打印总结
                final_state.update(updates)
        
        print("\n\n=== 进化完成 ===")
        print(f"总轮数: {final_state.get('generation_round')}")
        print(f"最佳上下文输出:\n{final_state.get('best_context')}")
        
        # 可选: 打印基因池统计
        print("\n--- 基因池统计 ---")
        if isinstance(final_state.get("gene_pool"), dict):
            for feature, chunks in final_state["gene_pool"].items():
                print(f"\n特征: {feature}")
                for chunk in chunks:
                    print(f"  ID: {chunk.id} | 使用次数: {chunk.usage_count} | 突变率: {chunk.mutation_rate:.2f} | 适应度贡献: {chunk.fitness_contribution:.2f}")
                    print(f"  内容: {chunk.content[:50]}...")
        elif final_state.get("gene_pool"):
             for chunk in final_state["gene_pool"]:
                print(f"ID: {chunk.id} | 使用次数: {chunk.usage_count} | 突变率: {chunk.mutation_rate:.2f} | 适应度贡献: {chunk.fitness_contribution:.2f}")
                print(f"内容: {chunk.content[:50]}...")
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
