"""传感器基础接口定义

传感器是感知层的核心组件，负责监听环境中的事件并将外部刺激转换为内部信号。
本模块定义了传感器的抽象基类和核心接口。

设计原则：
1. 事件驱动：传感器通过事件机制响应外部刺激
2. 异步处理：支持非阻塞的信号处理
3. 注意力调控：支持动态调整传感器的敏感度
4. 可扩展性：通过抽象基类支持多种传感器类型
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, AsyncGenerator, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime

# 类型定义
StimulusType = TypeVar('StimulusType')  # 外部刺激类型
SignalType = TypeVar('SignalType')      # 内部信号类型

@dataclass
class StimulusEvent:
    """外部刺激事件"""
    stimulus_id: str                    # 刺激唯一标识
    stimulus_type: str                  # 刺激类型
    data: Any                          # 刺激数据
    intensity: StimulusIntensity       # 刺激强度
    timestamp: datetime                # 时间戳
    source: str                        # 刺激源
    metadata: Dict[str, Any]           # 元数据


@dataclass
class InternalSignal:
    """内部信号"""
    signal_id: str                     # 信号唯一标识
    sensor_id: str                     # 产生信号的传感器ID
    signal_type: str                   # 信号类型
    data: Any                          # 信号数据
    confidence: float                  # 信号置信度 (0.0-1.0)
    attention_weight: float            # 注意力权重 (0.0-1.0)
    timestamp: datetime                # 时间戳
    processing_time: float             # 处理时间(毫秒)
    metadata: Dict[str, Any]           # 元数据


@dataclass
class AttentionConfig:
    """注意力配置"""
    sensitivity: float = 1.0           # 敏感度 (0.0-2.0)
    threshold: float = 0.1             # 阈值 (0.0-1.0)
    focus_weight: float = 1.0          # 聚焦权重 (0.0-2.0)
    decay_rate: float = 0.95           # 衰减率 (0.0-1.0)
    max_concurrent: int = 10           # 最大并发处理数


class ISensorEventHandler(ABC):
    """传感器事件处理器接口"""
    
    @abstractmethod
    async def on_stimulus_detected(self, event: StimulusEvent) -> None:
        """当检测到外部刺激时触发"""
        pass
    
    @abstractmethod
    async def on_signal_generated(self, signal: InternalSignal) -> None:
        """当生成内部信号时触发"""
        pass
    
    @abstractmethod
    async def on_sensor_state_changed(self, sensor_id: str, old_state: SensorState, new_state: SensorState) -> None:
        """当传感器状态改变时触发"""
        pass
    
    @abstractmethod
    async def on_attention_adjusted(self, sensor_id: str, old_config: AttentionConfig, new_config: AttentionConfig) -> None:
        """当注意力配置调整时触发"""
        pass


class BaseSensor(Generic[StimulusType, SignalType], ABC):
    """传感器抽象基类
    
    所有传感器都应该继承此基类并实现其抽象方法。
    传感器负责监听特定类型的外部刺激，并将其转换为内部信号。
    """
    
    def __init__(self, sensor_id: str, sensor_type: str):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.state = SensorState.INACTIVE
        self.attention_config = AttentionConfig()
        self.event_handlers: List[ISensorEventHandler] = []
        self.attention_regulator: Optional[IAttentionRegulator] = None
        self._stop_event = asyncio.Event()
        self._processing_tasks: List[asyncio.Task] = []
    
    # 抽象方法 - 子类必须实现
    @abstractmethod
    async def detect_stimulus(self) -> AsyncGenerator[StimulusEvent, None]:
        """检测外部刺激
        
        Returns:
            AsyncGenerator[StimulusEvent, None]: 异步生成器，产生检测到的刺激事件
        """
        pass
    
    @abstractmethod
    async def convert_to_signal(self, stimulus: StimulusEvent) -> InternalSignal:
        """将外部刺激转换为内部信号
        
        Args:
            stimulus: 外部刺激事件
            
        Returns:
            InternalSignal: 转换后的内部信号
        """
        pass
    
    @abstractmethod
    async def validate_stimulus(self, stimulus: StimulusEvent) -> bool:
        """验证刺激是否有效
        
        Args:
            stimulus: 外部刺激事件
            
        Returns:
            bool: 刺激是否有效
        """
        pass
    
    # 具体方法 - 提供默认实现
    async def start_listening(self) -> None:
        """开始监听外部刺激"""
        if self.state != SensorState.INACTIVE:
            raise RuntimeError(f"Sensor {self.sensor_id} is already active")
        
        await self._change_state(SensorState.ACTIVE)
        self._stop_event.clear()
        
        # 启动监听任务
        listen_task = asyncio.create_task(self._listen_loop())
        self._processing_tasks.append(listen_task)
        
        await self._change_state(SensorState.LISTENING)
    
    async def stop_listening(self) -> None:
        """停止监听"""
        self._stop_event.set()
        
        # 等待所有处理任务完成
        if self._processing_tasks:
            await asyncio.gather(*self._processing_tasks, return_exceptions=True)
            self._processing_tasks.clear()
        
        await self._change_state(SensorState.INACTIVE)
    
    async def pause_listening(self) -> None:
        """暂停监听"""
        if self.state == SensorState.LISTENING:
            await self._change_state(SensorState.PAUSED)
    
    async def resume_listening(self) -> None:
        """恢复监听"""
        if self.state == SensorState.PAUSED:
            await self._change_state(SensorState.LISTENING)
    
    async def update_attention_config(self, config: AttentionConfig) -> None:
        """更新注意力配置"""
        old_config = self.attention_config
        self.attention_config = config
        
        # 通知事件处理器
        for handler in self.event_handlers:
            await handler.on_attention_adjusted(self.sensor_id, old_config, config)
    
    def add_event_handler(self, handler: ISensorEventHandler) -> None:
        """添加事件处理器"""
        if handler not in self.event_handlers:
            self.event_handlers.append(handler)
    
    def remove_event_handler(self, handler: ISensorEventHandler) -> None:
        """移除事件处理器"""
        if handler in self.event_handlers:
            self.event_handlers.remove(handler)
    
    def set_attention_regulator(self, regulator: IAttentionRegulator) -> None:
        """设置注意力调控器"""
        self.attention_regulator = regulator
    
    # 私有方法
    async def _listen_loop(self) -> None:
        """监听循环"""
        try:
            async for stimulus in self.detect_stimulus():
                if self._stop_event.is_set():
                    break
                
                if self.state == SensorState.PAUSED:
                    await asyncio.sleep(0.1)  # 暂停时短暂等待
                    continue
                
                # 处理刺激
                process_task = asyncio.create_task(self._process_stimulus(stimulus))
                self._processing_tasks.append(process_task)
                
                # 清理已完成的任务
                self._processing_tasks = [task for task in self._processing_tasks if not task.done()]
                
                # 限制并发处理数量
                if len(self._processing_tasks) >= self.attention_config.max_concurrent:
                    await asyncio.sleep(0.01)  # 短暂等待
        
        except Exception as e:
            await self._change_state(SensorState.ERROR)
            raise e
    
    async def _process_stimulus(self, stimulus: StimulusEvent) -> None:
        """处理单个刺激"""
        try:
            await self._change_state(SensorState.PROCESSING)
            
            # 验证刺激
            if not await self.validate_stimulus(stimulus):
                return
            
            # 通知事件处理器
            for handler in self.event_handlers:
                await handler.on_stimulus_detected(stimulus)
            
            # 计算注意力权重
            attention_weight = 1.0
            if self.attention_regulator:
                attention_weight = await self.attention_regulator.calculate_attention_weight(
                    stimulus, self.attention_config
                )
            
            # 应用注意力阈值
            if attention_weight < self.attention_config.threshold:
                return  # 注意力权重过低，忽略此刺激
            
            # 转换为内部信号
            signal = await self.convert_to_signal(stimulus)
            signal.attention_weight = attention_weight
            
            # 通知事件处理器
            for handler in self.event_handlers:
                await handler.on_signal_generated(signal)
            
            await self._change_state(SensorState.LISTENING)
        
        except Exception as e:
            await self._change_state(SensorState.ERROR)
            raise e
    
    async def _change_state(self, new_state: SensorState) -> None:
        """改变传感器状态"""
        old_state = self.state
        self.state = new_state
        
        # 通知事件处理器
        for handler in self.event_handlers:
            await handler.on_sensor_state_changed(self.sensor_id, old_state, new_state)
    
    def get_status(self) -> Dict[str, Any]:
        """获取传感器状态信息"""
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type,
            "state": self.state.value,
            "attention_config": {
                "sensitivity": self.attention_config.sensitivity,
                "threshold": self.attention_config.threshold,
                "focus_weight": self.attention_config.focus_weight,
                "decay_rate": self.attention_config.decay_rate,
                "max_concurrent": self.attention_config.max_concurrent
            },
            "active_tasks": len(self._processing_tasks),
            "has_attention_regulator": self.attention_regulator is not None,
            "event_handlers_count": len(self.event_handlers)
        }