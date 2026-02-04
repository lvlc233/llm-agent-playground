import os
from pathlib import Path
from typing import List
from src.models import EnvironmentStep

class EnvironmentManager:
    """
    环境管理器
    逻辑：加载和管理多步骤环境场景
    目的：为Agent提供结构化的环境体验
    """
    def __init__(self, base_path: str = "data/environments"):
        self.base_path = Path(base_path)  # 环境文件的基础路径

    def load_scenario(self, scenario_name: str) -> List[EnvironmentStep]:
        """
        加载环境场景
        逻辑：读取指定场景文件夹中的所有txt文件，按文件名排序
        目的：构建有序的环境步骤序列
        """
        scenario_path = self.base_path / scenario_name
        if not scenario_path.exists():
            raise FileNotFoundError(f"场景 {scenario_name} 在 {scenario_path} 未找到")

        steps = []
        # 列出所有txt文件并按文件名排序
        files = sorted(scenario_path.glob("*.txt"), key=lambda p: p.name)
        
        for i, file_path in enumerate(files):
            with open(file_path, "r", encoding="utf-8") as f:
                raw_content = f.read()
            
            # 解析内容和裁判标准（使用 '---' 分隔）
            parts = raw_content.split("\n---\n", 1)
            content = parts[0].strip()
            rubric = parts[1].strip() if len(parts) > 1 else "未提供具体评分标准，请根据目标常识判断。"

            steps.append(EnvironmentStep(
                step_id=file_path.stem,
                content=content,
                rubric=rubric,
                order=i,
                file_path=str(file_path)
            ))
        
        return steps

    @staticmethod
    def get_initial_prompt(step_content: str, gene_prompt: str, energy: int) -> str:
        """
        生成初始提示词
        逻辑：组合基因提示、当前状态和环境信息
        目的：为Agent提供完整的上下文信息
        """
        return f"""
{gene_prompt}

[当前状态]
能量等级: {energy} (警告: 每个行动都会根据长度消耗能量。零能量 = 死亡。)

[环境]
{step_content}

[指令]
分析环境并决定你的下一步行动。
清晰地输出你的行动。
"""
