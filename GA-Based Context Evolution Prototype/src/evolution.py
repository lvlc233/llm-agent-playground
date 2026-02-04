import os
import time
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich.progress import track

from src.models import AgentConfig, Gene
from src.environment import EnvironmentManager
from src.simulation import run_simulation
from src.mutator import Mutator

console = Console()

class EvolutionEngine:
    def __init__(self, population_size: int = 4, generations: int = 3, log_dir: str = None):
        self.population_size = population_size
        self.generations = generations
        self.env_manager = EnvironmentManager()
        self.mutator = Mutator()
        self.population: List[AgentConfig] = []
        self.history: List[Dict] = [] # 记录每代的统计数据
        
        # 初始化日志目录
        if log_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.log_dir = os.path.join("logs", f"run_{timestamp}")
        else:
            self.log_dir = log_dir
        
        os.makedirs(self.log_dir, exist_ok=True)
        console.print(f"[blue]详细日志将保存在: {self.log_dir}[/blue]")

    def initialize_population(self):
        """初始化种群，创建多样化的初始Agent"""
        console.print("[bold blue]正在初始化种群...[/bold blue]")
        
        # 预定义的几个原型，确保初始多样性
        prototypes = [
            ("勇敢的探险家", "总是优先探索未知的路径，勇于尝试所有物品。"),
            ("谨慎的学者", "在行动前会仔细观察环境描述，分析物品的用途。"),
            ("鲁莽的寻宝者", "看到什么就拿什么，尝试用暴力解决问题。"),
            ("逻辑学家", "尝试推理环境中的因果关系，按步骤解决问题。")
        ]
        
        for i in range(self.population_size):
            proto_idx = i % len(prototypes)
            identity, strategy = prototypes[proto_idx]
            
            gene = Gene(
                identity=identity,
                strategy=strategy,
                memory=""
            )
            config = AgentConfig(gene=gene, generation=1)
            self.population.append(config)
            
        console.print(f"种群初始化完成，共 {len(self.population)} 个个体。")

    def run(self):
        """运行进化循环"""
        self.initialize_population()
        
        for gen in range(1, self.generations + 1):
            console.rule(f"[bold green]第 {gen} 代 / {self.generations}[/bold green]")
            
            # 1. 评估当前种群
            results = self.evaluate_population(gen)
            
            # 2. 统计与展示
            self.display_generation_stats(results, gen)
            
            # 3. 记录历史
            best_fitness = max(r["fitness"] for r in results)
            avg_fitness = sum(r["fitness"] for r in results) / len(results)
            self.history.append({
                "generation": gen,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
                "best_agent": next(r for r in results if r["fitness"] == best_fitness)
            })
            
            # 如果是最后一代，不需要繁衍
            if gen < self.generations:
                self.population = self.breed_next_generation(results, gen)
                
        self.display_final_report()

    def evaluate_population(self, generation: int) -> List[Dict]:
        """评估种群中每个个体的适应度"""
        results = []
        
        # 创建本代日志目录
        gen_dir = os.path.join(self.log_dir, f"gen_{generation}")
        os.makedirs(gen_dir, exist_ok=True)
        
        # 串行运行（未来可以用并行优化）
        for i, agent_config in enumerate(track(self.population, description=f"评估第 {generation} 代...")):
            # 更新代数
            agent_config.generation = generation
            
            # 运行模拟
            sim_result = run_simulation(agent_config, self.env_manager)
            results.append(sim_result)
            
            # 保存详细日志
            self.save_agent_log(gen_dir, sim_result)
            
        return results

    def save_agent_log(self, gen_dir: str, result: Dict):
        """将Agent的运行日志保存为Markdown文件"""
        agent_id = result["agent_id"]
        filename = os.path.join(gen_dir, f"{agent_id}.md")
        
        gene = result["gene"]
        logs = result.get("logs", [])
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# Agent {agent_id} 模拟报告\n\n")
            f.write(f"- **适应度**: {result['fitness']}\n")
            f.write(f"- **解决步数**: {result['solved_steps_count']}\n")
            f.write(f"- **剩余能量**: {result['final_energy']}\n")
            f.write(f"- **死因**: {result.get('cause_of_death', '无')}\n\n")
            
            f.write("## 基因图谱\n")
            f.write(f"### 身份\n{gene.identity}\n")
            f.write(f"### 策略\n{gene.strategy}\n")
            f.write(f"### 记忆\n{gene.memory}\n\n")
            
            f.write("## 交互日志\n")
            for entry in logs:
                step = entry.get("step")
                log_type = entry.get("type")
                content = entry.get("content")
                meta = entry.get("metadata", {})
                
                if log_type == "perception":
                    f.write(f"### Step {step}: 环境感知\n")
                    f.write(f"> **当前区域**: {meta.get('step_index')}\n\n")
                    f.write(f"```text\n{content}\n```\n\n")
                    
                elif log_type == "agent":
                    f.write(f"### Step {step}: Agent行动\n")
                    f.write(f"**能量**: {meta.get('energy')}\n\n")
                    f.write(f"{content}\n\n")
                    
                elif log_type == "judge":
                    f.write(f"### Step {step}: 裁判反馈\n")
                    f.write(f"**解决**: {meta.get('is_solved')} | **奖励**: {meta.get('energy_reward')}\n\n")
                    f.write(f"> {content}\n\n")
                    f.write("---\n\n")

    def breed_next_generation(self, results: List[Dict], current_gen: int) -> List[AgentConfig]:
        """繁衍下一代：精英保留 + 变异交叉"""
        # 按适应度排序
        sorted_results = sorted(results, key=lambda x: x["fitness"], reverse=True)
        
        next_gen_configs = []
        
        # 1. 精英保留 (Elitism): 保留最好的1个
        elite = sorted_results[0]
        console.print(f"[yellow]精英保留:[/yellow] {elite['agent_id']} (Fitness: {elite['fitness']})")
        next_gen_configs.append(AgentConfig(
            gene=elite["gene"],
            generation=current_gen + 1
        ))
        
        # 2. 繁殖填补剩余空位
        while len(next_gen_configs) < self.population_size:
            # 简单的锦标赛选择
            parent_a_res = self.tournament_select(sorted_results)
            parent_b_res = self.tournament_select(sorted_results)
            
            # 交叉变异
            child_gene = self.mutator.evolve(
                parent_a_res["gene"], 
                parent_b_res["gene"],
                parent_a_res["fitness"],
                parent_b_res["fitness"],
                parent_a_id=parent_a_res["agent_id"],
                parent_b_id=parent_b_res["agent_id"]
            )
            
            next_gen_configs.append(AgentConfig(
                gene=child_gene,
                generation=current_gen + 1
            ))
            
        return next_gen_configs

    def tournament_select(self, sorted_results: List[Dict], k=2) -> Dict:
        """锦标赛选择"""
        import random
        candidates = random.sample(sorted_results, k=min(k, len(sorted_results)))
        return max(candidates, key=lambda x: x["fitness"])

    def display_generation_stats(self, results: List[Dict], generation: int):
        """展示每代的统计信息"""
        table = Table(title=f"第 {generation} 代 评估结果")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("适应度", style="magenta")
        table.add_column("解决步数", style="green")
        table.add_column("剩余能量", style="yellow")
        table.add_column("策略摘要", style="white")
        
        # 按适应度排序
        sorted_results = sorted(results, key=lambda x: x["fitness"], reverse=True)
        
        for res in sorted_results:
            strategy_summary = res["gene"].strategy[:30] + "..." if len(res["gene"].strategy) > 30 else res["gene"].strategy
            table.add_row(
                res["agent_id"],
                str(res["fitness"]),
                str(res["solved_steps_count"]),
                str(res["final_energy"]),
                strategy_summary
            )
            
        console.print(table)

    def display_final_report(self):
        """展示最终进化报告"""
        console.rule("[bold red]进化完成报告[/bold red]")
        
        # 打印历史趋势
        console.print("\n[进化趋势]")
        for entry in self.history:
            gen = entry["generation"]
            best = entry["best_fitness"]
            avg = entry["avg_fitness"]
            console.print(f"Gen {gen}: Best={best}, Avg={avg:.1f}")
            
        # 展示最终最佳个体
        best_ever = max(self.history, key=lambda x: x["best_fitness"])
        best_agent_res = best_ever["best_agent"]
        
        console.print(Panel(
            best_agent_res["gene"].to_prompt_string(),
            title=f"🏆 史上最强个体 (Gen {best_ever['generation']}, Fitness {best_ever['best_fitness']})",
            border_style="gold1"
        ))

from rich.panel import Panel
