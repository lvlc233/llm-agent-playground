"""
测试 Literal 中使用字典 keys 的不同方法
"""

from typing import Literal
from pydantic import BaseModel, Field

# 原始数据结构
meta_thinks = [
    {"Association": "one idea hooks another, branch after branch, no hurry to converge."},
    {"Deduction": "take a premise you trust and push it to its 'then,' watching for absurdity."},
    {"Induction": "pile the scattered cases together, sketch a contour, ready for the next black swan."},
    {"Analogy": "lift the skeleton of A onto B, letting familiarity light up the strange—while remembering 'similar' is not 'same.'"},
    {"Counterfactual": "rewind the timeline, ask 'if only…,' to probe which causal link snaps first."},
    {"Externalization": "off-load the thought onto paper, model, or conversation; let the eyes take over from working memory and push the symbol-loop further."},
    {"Reflection": "turn the camera on the camera-operator—ask 'why did I just believe that? what lets me doubt it?' The only move that can edit the other six."},
]

# 方法1: 提取 keys 并使用 * 展开（推荐）
meta_think_types = tuple(list(think_dict.keys())[0] for think_dict in meta_thinks)
print("提取的 keys:", meta_think_types)

class ThinkSelect1(BaseModel):
    think_type: Literal[*meta_think_types] = Field(description="选择思考类型")

# 方法2: 直接定义（如果你想要更明确）
class ThinkSelect2(BaseModel):
    think_type: Literal[
        "Association", 
        "Deduction", 
        "Induction", 
        "Analogy", 
        "Counterfactual", 
        "Externalization", 
        "Reflection"
    ] = Field(description="选择思考类型")

# 方法3: 使用 Union（另一种写法）
from typing import Union

ThinkType = Union[
    Literal["Association"],
    Literal["Deduction"], 
    Literal["Induction"],
    Literal["Analogy"],
    Literal["Counterfactual"],
    Literal["Externalization"],
    Literal["Reflection"]
]

class ThinkSelect3(BaseModel):
    think_type: ThinkType = Field(description="选择思考类型")

# 测试函数
def test_literal_validation():
    """测试 Literal 类型验证"""
    
    print("=== 测试方法1: 动态提取 keys ===")
    try:
        # 有效值
        valid_choice = ThinkSelect1(think_type="Association")
        print(f"✅ 有效选择: {valid_choice.think_type}")
        
        # 无效值
        try:
            invalid_choice = ThinkSelect1(think_type="InvalidType")
        except Exception as e:
            print(f"❌ 无效选择被正确拒绝: {e}")
            
    except Exception as e:
        print(f"❌ 方法1 错误: {e}")
    
    print("\n=== 测试方法2: 直接定义 ===")
    try:
        valid_choice2 = ThinkSelect2(think_type="Deduction")
        print(f"✅ 有效选择: {valid_choice2.think_type}")
    except Exception as e:
        print(f"❌ 方法2 错误: {e}")
    
    print("\n=== 测试方法3: Union 类型 ===")
    try:
        valid_choice3 = ThinkSelect3(think_type="Reflection")
        print(f"✅ 有效选择: {valid_choice3.think_type}")
    except Exception as e:
        print(f"❌ 方法3 错误: {e}")

def get_think_description(think_type: str) -> str:
    """根据思考类型获取描述"""
    for think_dict in meta_thinks:
        if think_type in think_dict:
            return think_dict[think_type]
    return "未找到描述"

def demo_usage():
    """演示实际使用"""
    print("\n=== 实际使用演示 ===")
    
    # 创建选择
    choice = ThinkSelect1(think_type="Analogy")
    print(f"选择的思考类型: {choice.think_type}")
    
    # 获取对应描述
    description = get_think_description(choice.think_type)
    print(f"描述: {description}")
    
    # 显示所有可用选项
    print(f"\n所有可用的思考类型: {list(meta_think_types)}")

if __name__ == "__main__":
    test_literal_validation()
    demo_usage()