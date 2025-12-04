import time
import csv
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class 实验结果:
    实验名称: str
    方法名称: str
    输入token数: int
    输出token数: int
    总token数: int
    响应延迟: float
    成功率: bool
    错误信息: str = None
    输出内容: Any = None
    时间戳: str = None
    模型名称: str = None

class 中文指标追踪器:
    def __init__(self, 实验名称: str = "未命名实验", 模型名称: str = "未知模型"):
        self.实验名称 = 实验名称
        self.模型名称 = 模型名称
        self.结果列表: List[实验结果] = []
        self.开始时间 = datetime.now()

        # 创建结果保存目录
        self.结果目录 = "实验结果"
        if not os.path.exists(self.结果目录):
            os.makedirs(self.结果目录)

    def 记录结果(self, 方法名称: str, 响应对象: Any, 开始时间: float,
              成功率: bool = True, 错误信息: str = None, 解析输出: Any = None):
        延迟 = time.time() - 开始时间

        输入token数 = 0
        输出token数 = 0

        # 从响应中提取token使用量
        if hasattr(响应对象, 'usage_metadata') and 响应对象.usage_metadata:
            输入token数 = 响应对象.usage_metadata.get('input_tokens', 0)
            输出token数 = 响应对象.usage_metadata.get('output_tokens', 0)
        elif hasattr(响应对象, 'response_metadata'):
            token使用情况 = 响应对象.response_metadata.get('token_usage', {})
            输入token数 = token使用情况.get('prompt_tokens', 0)
            输出token数 = token使用情况.get('completion_tokens', 0)

        结果 = 实验结果(
            实验名称=self.实验名称,
            方法名称=方法名称,
            输入token数=输入token数,
            输出token数=输出token数,
            总token数=输入token数 + 输出token数,
            响应延迟=延迟,
            成功率=成功率,
            错误信息=错误信息,
            输出内容=解析输出,
            时间戳=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            模型名称=self.模型名称
        )

        self.结果列表.append(结果)

    def 打印中文摘要(self):
        """打印中文实验摘要"""
        print(f"\n{'='*20} 实验摘要 {'='*20}")
        print(f"实验名称: {self.实验名称}")
        print(f"模型名称: {self.模型名称}")
        print(f"测试时间: {self.开始时间.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'方法':<15} | {'输入Token':<10} | {'输出Token':<10} | {'延迟(秒)':<10} | {'状态':<8}")
        print("-" * 65)
        for 结果 in self.结果列表:
            状态 = "成功" if 结果.成功率 else "失败"
            print(f"{结果.方法名称:<15} | {结果.输入token数:<10} | {结果.输出token数:<10} | {结果.响应延迟:<10.4f} | {状态:<8}")
        print("=" * 65)

    def 保存CSV文件(self, 文件名: str = None):
        """保存结果为CSV文件"""
        if not 文件名:
            时间戳 = datetime.now().strftime("%Y%m%d_%H%M%S")
            文件名 = f"{self.实验名称}_{时间戳}.csv"

        文件路径 = os.path.join(self.结果目录, 文件名)

        with open(文件路径, 'w', newline='', encoding='utf-8') as csv文件:
            字段名 = ['实验名称', '方法名称', '模型名称', '输入token数', '输出token数',
                     '总token数', '响应延迟', '成功率', '错误信息', '时间戳']

            writer = csv.DictWriter(csv文件, fieldnames=字段名)
            writer.writeheader()

            for 结果 in self.结果列表:
                writer.writerow({
                    '实验名称': 结果.实验名称,
                    '方法名称': 结果.方法名称,
                    '模型名称': 结果.模型名称,
                    '输入token数': 结果.输入token数,
                    '输出token数': 结果.输出token数,
                    '总token数': 结果.总token数,
                    '响应延迟': 结果.响应延迟,
                    '成功率': 结果.成功率,
                    '错误信息': 结果.错误信息 or '',
                    '时间戳': 结果.时间戳
                })

        print(f"\n📊 实验结果已保存到: {文件路径}")
        return 文件路径

    def 获取统计信息(self) -> Dict[str, Any]:
        """获取实验统计信息"""
        if not self.结果列表:
            return {}

        成功次数 = sum(1 for 结果 in self.结果列表 if 结果.成功率)
        总次数 = len(self.结果列表)

        return {
            '实验名称': self.实验名称,
            '模型名称': self.模型名称,
            '总测试次数': 总次数,
            '成功次数': 成功次数,
            '失败次数': 总次数 - 成功次数,
            '成功率': f"{(成功次数/总次数)*100:.1f}%",
            '平均输入token数': sum(结果.输入token数 for 结果 in self.结果列表) / 总次数,
            '平均输出token数': sum(结果.输出token数 for 结果 in self.结果列表) / 总次数,
            '平均延迟': sum(结果.响应延迟 for 结果 in self.结果列表) / 总次数,
            '总token消耗': sum(结果.总token数 for 结果 in self.结果列表)
        }

    def 打印详细统计(self):
        """打印详细的统计信息"""
        统计 = self.获取统计信息()
        if not 统计:
            print("没有实验数据")
            return

        print(f"\n📈 详细统计信息")
        print(f"实验名称: {统计['实验名称']}")
        print(f"模型名称: {统计['模型名称']}")
        print(f"总测试次数: {统计['总测试次数']}")
        print(f"成功次数: {统计['成功次数']}")
        print(f"失败次数: {统计['失败次数']}")
        print(f"成功率: {统计['成功率']}")
        print(f"平均输入token数: {统计['平均输入token数']:.0f}")
        print(f"平均输出token数: {统计['平均输出token数']:.0f}")
        print(f"平均延迟: {统计['平均延迟']:.3f}秒")
        print(f"总token消耗: {统计['总token消耗']}")


# 原有的英文类保持兼容性，但内部使用中文
class MetricsTracker(中文指标追踪器):
    """兼容原有的英文接口"""
    def __init__(self, experiment_name: str = "Unnamed Experiment", model_name: str = "Unknown Model"):
        super().__init__(实验名称=experiment_name, 模型名称=model_name)

    def record(self, method_name: str, response: Any, start_time: float,
              success: bool = True, error: str = None, parsed_output: Any = None):
        super().记录结果(方法名称, response, start_time, success, error, parsed_output)

    def print_summary(self):
        super().打印中文摘要()

    def save_to_csv(self, filename: str = None):
        return super().保存CSV文件(filename)

    def get_statistics(self):
        return super().获取统计信息()

    def print_detailed_stats(self):
        super().打印详细统计()