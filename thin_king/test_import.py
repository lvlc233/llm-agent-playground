#!/usr/bin/env python3
"""
测试脚本：验证所有实验版本的导入和基本结构
"""

def test_imports():
    """测试所有版本的导入"""
    print("=== 测试导入 ===")
    
    try:
        from experiments import graphv1, graphv2, graphv3, graphv4, graphv5
        print("✅ 所有版本导入成功!")
        
        # 检查每个图的基本属性
        graphs = {
            "v1": graphv1,
            "v2": graphv2, 
            "v3": graphv3,
            "v4": graphv4,
            "v5": graphv5
        }
        
        for version, graph in graphs.items():
            print(f"✅ {version}: {type(graph).__name__}")
            
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    
    return True

def test_common_modules():
    """测试共享模块"""
    print("\n=== 测试共享模块 ===")
    
    try:
        from common.config import ModelConfig
        from common.tools import basic_thinking_tools, all_thinking_tools
        from common.prompts import BASE_THINKING_PROMPT, TOOL_THINKING_PROMPT
        from common.utils import ThinkingUtils, ExperimentUtils
        
        print("✅ 共享模块导入成功!")
        print(f"✅ 基础思考工具数量: {len(basic_thinking_tools)}")
        print(f"✅ 所有思考工具数量: {len(all_thinking_tools)}")
        
    except ImportError as e:
        print(f"❌ 共享模块导入失败: {e}")
        return False
    
    return True

def test_structure():
    """测试项目结构"""
    print("\n=== 测试项目结构 ===")
    
    import os
    
    # 检查关键目录和文件
    required_paths = [
        "common/__init__.py",
        "common/config.py",
        "common/tools.py", 
        "common/prompts.py",
        "common/utils.py",
        "experiments/__init__.py",
        "experiments/v1/__init__.py",
        "experiments/v1/agent.py",
        "experiments/v1/README.md",
        "experiments/v2/__init__.py",
        "experiments/v2/agent.py", 
        "experiments/v2/README.md",
        "experiments/v3/__init__.py",
        "experiments/v3/agent.py",
        "experiments/v3/README.md",
        "experiments/v4/__init__.py",
        "experiments/v4/agent.py",
        "experiments/v4/README.md",
        "experiments/v5/__init__.py",
        "experiments/v5/agent.py",
        "experiments/v5/README.md",
        "main.py",
        "README.md"
    ]
    
    missing_files = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_files.append(path)
    
    if missing_files:
        print("❌ 缺少以下文件:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print("✅ 所有必需文件都存在!")
        return True

def main():
    """主测试函数"""
    print("Thin_King 项目结构测试")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_common_modules, 
        test_structure
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    if all(results):
        print("🎉 所有测试通过! 项目重构成功!")
    else:
        print("❌ 部分测试失败，请检查项目结构")
    
    return all(results)

if __name__ == "__main__":
    main()