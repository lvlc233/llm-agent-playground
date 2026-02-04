import os
import json
import random
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.models import Gene

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

AGENT_MODEL_NAME = os.getenv("AGENT_MODEL_NAME", "deepseek-ai/DeepSeek-V3")
console = Console()

class Mutator:
    """
    负责基因的交叉（Crossover）和变异（Mutation）
    逻辑：利用LLM作为变异算子，结合两个父代基因生成新的子代
    """
    def __init__(self):
        self.llm = ChatOpenAI(
            model=AGENT_MODEL_NAME, 
            temperature=0.9, # 高温以增加多样性
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        
    def evolve(self, parent_a: Gene, parent_b: Gene, fitness_a: float, fitness_b: float, parent_a_id: str = "A", parent_b_id: str = "B") -> Gene:
        """
        进化操作：结合两个父代生成子代
        """
        # 简单模拟模式（如果没有API Key）
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key.startswith("sk-proj-xxxx") or not api_key:
            return self._mock_evolve(parent_a, parent_b)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位精通进化算法的AI遗传工程师。
你的任务是根据两个父代Agent的基因，创造一个新的子代Agent基因。
父代基因由[身份]、[策略]和[记忆]组成。

请遵循以下原则：
1. **优胜劣汰**：倾向于保留表现更好的父代（Fitness更高）的特征。
2. **交叉重组**：混合两个父代的策略和身份特征，创造逻辑自洽的新组合。
3. **基因突变**：在策略中引入微小的、实验性的改变（变异），以探索新的可能性。不要只是复制。
4. **记忆传承**：将父代最关键的成功经验总结为简短的[记忆]传给子代。

输出必须是严格的JSON格式：
{{
  "reasoning": "分析父代优缺点，解释本次突变和交叉的逻辑...",
  "identity": "新的身份描述...",
  "strategy": "新的策略描述...",
  "memory": "新的记忆描述..."
}}
"""),
            ("user", """
Parent A (ID: {id_a}, Fitness: {fitness_a}):
{gene_a}

Parent B (ID: {id_b}, Fitness: {fitness_b}):
{gene_b}

请生成 Child Gene (JSON):
""")
        ])
        
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            result = chain.invoke({
                "id_a": parent_a_id,
                "fitness_a": fitness_a,
                "gene_a": parent_a.to_prompt_string(),
                "id_b": parent_b_id,
                "fitness_b": fitness_b,
                "gene_b": parent_b.to_prompt_string()
            })
            
            # 可视化进化过程
            self._visualize_mutation(parent_a_id, fitness_a, parent_b_id, fitness_b, result)
            
            return Gene(
                identity=result.get("identity", parent_a.identity),
                strategy=result.get("strategy", parent_a.strategy),
                memory=result.get("memory", "")
            )
            
        except Exception as e:
            console.print(f"[red]Mutator Error:[/red] {e}")
            # Fallback: return a random parent with slight noise
            return self._mock_evolve(parent_a, parent_b)

    def _visualize_mutation(self, id_a, fit_a, id_b, fit_b, result):
        """在控制台打印进化详情"""
        tree = Tree(f"🧬 [bold magenta]基因进化发生[/bold magenta]")
        
        parents = tree.add("👪 父母")
        parents.add(f"[cyan]{id_a}[/cyan] (Fit: {fit_a})")
        parents.add(f"[cyan]{id_b}[/cyan] (Fit: {fit_b})")
        
        logic = tree.add("🧠 [yellow]进化逻辑 (Reasoning)[/yellow]")
        logic.add(f"[italic]{result.get('reasoning', '无')}[/italic]")
        
        child = tree.add("👶 [green]新子代[/green]")
        child.add(f"身份: {result.get('identity')}")
        child.add(f"策略: {result.get('strategy')[:100]}...")
        child.add(f"记忆: {result.get('memory')}")
        
        console.print(Panel(tree, border_style="magenta", expand=False))

    def _mock_evolve(self, parent_a: Gene, parent_b: Gene) -> Gene:
        """无LLM时的模拟进化"""
        import random
        base = parent_a if random.random() > 0.5 else parent_b
        return Gene(
            identity=base.identity + f" (v{random.randint(1,9)})",
            strategy=base.strategy + " [Mutation]",
            memory=base.memory
        )
