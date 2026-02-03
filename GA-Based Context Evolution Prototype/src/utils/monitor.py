import sys
from typing import Any, Dict, List
from loguru import logger

# 配置 Loguru
logger.remove()
# 添加新的 handler，输出到 stdout
logger.add(
    sys.stdout, 
    format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
    level="INFO",
    colorize=True
)

class StateMonitor:
    """
    使用 Loguru 进行结构化、高亮显示的 LangGraph 状态监控。
    支持详细展示基因池、染色体组合及上下文内容。
    """
    
    @staticmethod
    def _log(message: str):
        """Helper to log with color parsing enabled."""
        logger.opt(colors=True).info(message)

    @staticmethod
    def print_step(node_name: str, updates: Dict[str, Any]):
        """
        打印节点的更新信息。
        """
        StateMonitor._log(f"\n{'='*20} 🟢 节点完成: {node_name} {'='*20}")
        
        for key, value in updates.items():
            StateMonitor._log_key_value(key, value)
            
        StateMonitor._log(f"{'='*60}\n")

    @staticmethod
    def _log_key_value(key: str, value: Any, indent_level: int = 0):
        indent = "  " * indent_level
        
        # 1. 基因池 (Gene Pool) 特殊处理
        if key == "gene_pool" and isinstance(value, dict):
            StateMonitor._log(f"{indent}<cyan>🧬 基因池 ({key})</cyan>:")
            for feature, chunks in value.items():
                StateMonitor._log(f"{indent}  <yellow>特征槽位: {feature}</yellow> (包含 {len(chunks)} 个基因片段)")
                for chunk in chunks:
                    # 兼容 Pydantic 对象和字典
                    chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else (chunk.dict() if hasattr(chunk, "dict") else chunk)
                    
                    cid = chunk_dict.get("id", "N/A")
                    content = chunk_dict.get("content", "")
                    usage = chunk_dict.get("usage_count", 0)
                    
                    # 格式化内容，支持多行缩进
                    content_preview = content.replace("\n", f"\n{indent}      ")
                    
                    StateMonitor._log(f"{indent}    - <blue>ID: [{cid}]</blue> (引用次数: {usage})")
                    StateMonitor._log(f"{indent}      <dim>内容: {content_preview}</dim>")

        # 2. 染色体 (Chromosomes) 特殊处理
        elif key == "chromosomes" and isinstance(value, list):
            StateMonitor._log(f"{indent}<magenta>🐛 种群染色体 ({key})</magenta> (共 {len(value)} 个个体):")
            for i, chromo in enumerate(value):
                c_data = chromo.model_dump() if hasattr(chromo, "model_dump") else (chromo.dict() if hasattr(chromo, "dict") else chromo)
                cid = c_data.get("id")
                chunk_ids = c_data.get("chunk_ids", [])
                StateMonitor._log(f"{indent}  [{i}] <bold>ID: {cid}</bold>")
                StateMonitor._log(f"{indent}      基因序列: {chunk_ids}")

        # 3. 评估结果 (Evaluation Results) 特殊处理
        elif key == "evaluation_results" and isinstance(value, list):
            StateMonitor._log(f"{indent}<green>📊 评估结果 ({key})</green>:")
            for res in value:
                r_data = res.model_dump() if hasattr(res, "model_dump") else (res.dict() if hasattr(res, "dict") else res)
                cid = r_data.get("chromosome_id")
                score = r_data.get("fitness_score", 0)
                reason = r_data.get("reasoning", "")
                context = r_data.get("generated_context", "")
                
                actual_features = r_data.get("actual_features", [])
                
                # 根据分数变色
                score_color = "green" if score >= 80 else "yellow" if score >= 50 else "red"
                
                StateMonitor._log(f"{indent}  <bold>染色体: {cid}</bold> | 得分: <{score_color}>{score:.2f}</{score_color}>")
                StateMonitor._log(f"{indent}    实际特征: {actual_features}")
                StateMonitor._log(f"{indent}    评价: {reason}")
                
                # 显示完整内容
                StateMonitor._log(f"{indent}    <white>生成上下文 (Full Content):</white>")
                StateMonitor._log(f"{indent}    " + "-"*40)
                formatted_context = context.replace("\n", f"\n{indent}    ")
                StateMonitor._log(f"{indent}    {formatted_context}")
                StateMonitor._log(f"{indent}    " + "-"*40)

        # 4. 理想画像 (Ideal Profile)
        elif key == "ideal_context_profile" and isinstance(value, list):
             StateMonitor._log(f"{indent}<cyan>🎯 理想画像特征 ({key})</cyan>:")
             for feature in value:
                 StateMonitor._log(f"{indent}  - {feature}")

        # 5. 最佳结果 (Best Context)
        elif key == "best_context" and isinstance(value, str):
            StateMonitor._log(f"{indent}<red>🏆 最佳上下文 ({key})</red>:")
            StateMonitor._log(f"{indent}    " + "-"*40)
            formatted_context = value.replace("\n", f"\n{indent}    ")
            StateMonitor._log(f"{indent}    {formatted_context}")
            StateMonitor._log(f"{indent}    " + "-"*40)

        # 6. 其他列表
        elif isinstance(value, list):
            StateMonitor._log(f"{indent}<white>📂 {key}</white> [{len(value)} items]:")
            for item in value[:5]: # 限制显示数量，避免刷屏
                StateMonitor._log(f"{indent}  - {str(item)[:100]}")
            if len(value) > 5:
                StateMonitor._log(f"{indent}  ... (还有 {len(value)-5} 项)")

        # 7. 其他字典
        elif isinstance(value, dict):
             StateMonitor._log(f"{indent}<white>📂 {key}</white>:")
             for k, v in value.items():
                 StateMonitor._log_key_value(k, v, indent_level + 1)

        # 8. 基础类型
        else:
            val_str = str(value)
            if len(val_str) > 200:
                 val_str = val_str[:200] + "..."
            StateMonitor._log(f"{indent}<white>🔹 {key}</white>: {val_str}")
