import time
import csv
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class ExperimentResult:
    method_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency: float
    success: bool
    error: str = None
    output: Any = None

class MetricsTracker:
    def __init__(self, experiment_name: str = "Unnamed Experiment", model_name: str = "Unknown Model"):
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.results: List[ExperimentResult] = []
        self.start_time = datetime.now()

        # 创建结果保存目录
        self.results_dir = "实验结果"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def record(self, method_name: str, response: Any, start_time: float, success: bool = True, error: str = None, parsed_output: Any = None):
        latency = time.time() - start_time
        
        input_tokens = 0
        output_tokens = 0
        
        # Attempt to extract token usage from response
        # This varies by integration, but for ChatOpenAI it's usually in usage_metadata
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            input_tokens = response.usage_metadata.get('input_tokens', 0)
            output_tokens = response.usage_metadata.get('output_tokens', 0)
        elif hasattr(response, 'response_metadata'):
             token_usage = response.response_metadata.get('token_usage', {})
             input_tokens = token_usage.get('prompt_tokens', 0)
             output_tokens = token_usage.get('completion_tokens', 0)

        self.results.append(ExperimentResult(
            method_name=method_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency=latency,
            success=success,
            error=error,
            output=parsed_output
        ))

    def print_summary(self):
        """打印中文实验摘要"""
        print(f"\n{'='*20} 实验摘要 {'='*20}")
        print(f"实验名称: {self.experiment_name}")
        print(f"模型名称: {self.model_name}")
        print(f"测试时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'方法':<20} | {'输入Token':<10} | {'输出Token':<10} | {'延迟(秒)':<10} | {'状态':<8}")
        print("-" * 70)
        for res in self.results:
            status = "成功" if res.success else "失败"
            print(f"{res.method_name:<20} | {res.input_tokens:<10} | {res.output_tokens:<10} | {res.latency:<10.4f} | {status:<8}")
        print("=" * 70)

    def save_to_csv(self, filename: str = None):
        """保存结果为CSV文件"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.experiment_name}_{timestamp}.csv"

        # Sanitize filename to remove illegal characters
        import re
        filename = re.sub(r'[\\/*?:"<>|]', '_', filename)

        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Experiment Name', 'Method Name', 'Model Name', 'Input Tokens', 'Output Tokens',
                         'Total Tokens', 'Latency (Seconds)', 'Success', 'Error Message', 'Test Time']

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for res in self.results:
                writer.writerow({
                    'Experiment Name': self.experiment_name,
                    'Method Name': res.method_name,
                    'Model Name': self.model_name,
                    'Input Tokens': res.input_tokens,
                    'Output Tokens': res.output_tokens,
                    'Total Tokens': res.total_tokens,
                    'Latency (Seconds)': res.latency,
                    'Success': 'Success' if res.success else 'Failed',
                    'Error Message': res.error or '',
                    'Test Time': self.start_time.strftime('%Y-%m-%d %H:%M:%S')
                })

        print(f"\n实验结果已保存到: {filepath}")
        return filepath

    def get_statistics(self):
        """获取实验统计信息"""
        if not self.results:
            return {}

        successful_count = sum(1 for res in self.results if res.success)
        total_count = len(self.results)

        return {
            'Experiment Name': self.experiment_name,
            'Model Name': self.model_name,
            'Total Tests': total_count,
            'Successful': successful_count,
            'Failed': total_count - successful_count,
            'Success Rate': f"{(successful_count/total_count)*100:.1f}%",
            'Average Input Tokens': sum(res.input_tokens for res in self.results) / total_count,
            'Average Output Tokens': sum(res.output_tokens for res in self.results) / total_count,
            'Average Latency (Seconds)': sum(res.latency for res in self.results) / total_count,
            'Total Token Consumption': sum(res.total_tokens for res in self.results)
        }

    def print_detailed_stats(self):
        """打印详细的统计信息"""
        stats = self.get_statistics()
        if not stats:
            print("没有实验数据")
            return

        print(f"\n详细统计信息")
        print(f"实验名称: {stats['Experiment Name']}")
        print(f"模型名称: {stats['Model Name']}")
        print(f"总测试次数: {stats['Total Tests']}")
        print(f"成功次数: {stats['Successful']}")
        print(f"失败次数: {stats['Failed']}")
        print(f"成功率: {stats['Success Rate']}")
        print(f"平均输入Token数: {stats['Average Input Tokens']:.0f}")
        print(f"平均输出Token数: {stats['Average Output Tokens']:.0f}")
        print(f"平均延迟: {stats['Average Latency (Seconds)']:.3f}秒")
        print(f"总Token消耗: {stats['Total Token Consumption']}")
