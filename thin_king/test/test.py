import json
from typing import Any, Optional, List
from uuid import UUID
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


class LLMDebugCallbackHandler(BaseCallbackHandler):
    """自定义回调处理器，用于观察发送给LLM的详细信息"""
    
    def __init__(self, verbose: bool = True):
        """
        初始化回调处理器
        
        Args:
            verbose: 是否打印详细信息
        """
        self.verbose = verbose
    
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """当聊天模型开始运行时调用，可以观察发送给LLM的所有信息"""
        if self.verbose:
            print("\n" + "="*80)
            print("🚀 LLM 调用开始")
            print("="*80)
            
            # 打印序列化信息（包含模型配置）
            print("\n📋 模型配置信息:")
            print(json.dumps(serialized, indent=2, ensure_ascii=False))
            
            # 打印消息内容
            print("\n💬 发送给LLM的消息:")
            for i, message_batch in enumerate(messages):
                print(f"\n--- 消息批次 {i+1} ---")
                for j, message in enumerate(message_batch):
                    print(f"消息 {j+1}: {type(message).__name__}")
                    print(f"内容: {message.content}")
                    
                    # 如果消息有工具调用信息
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        print(f"工具调用: {json.dumps(message.tool_calls, indent=2, ensure_ascii=False)}")
                    
                    # 如果消息有额外的kwargs
                    if hasattr(message, 'additional_kwargs') and message.additional_kwargs:
                        print(f"额外参数: {json.dumps(message.additional_kwargs, indent=2, ensure_ascii=False)}")
            
            # 打印运行时信息
            print(f"\n🔍 运行信息:")
            print(f"Run ID: {run_id}")
            print(f"Parent Run ID: {parent_run_id}")
            print(f"Tags: {tags}")
            print(f"Metadata: {metadata}")
            
            # 打印其他kwargs（可能包含工具信息）
            if kwargs:
                print(f"\n🛠️ 其他参数:")
                print(json.dumps(kwargs, indent=2, ensure_ascii=False, default=str))
            
            print("="*80)
    
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """当普通LLM开始运行时调用"""
        if self.verbose:
            print("\n" + "="*80)
            print("🚀 LLM 调用开始 (非聊天模型)")
            print("="*80)
            
            print("\n📋 模型配置信息:")
            print(json.dumps(serialized, indent=2, ensure_ascii=False))
            
            print("\n📝 提示词:")
            for i, prompt in enumerate(prompts):
                print(f"提示词 {i+1}: {prompt}")
            
            print(f"\n🔍 运行信息:")
            print(f"Run ID: {run_id}")
            print(f"Parent Run ID: {parent_run_id}")
            print(f"Tags: {tags}")
            print(f"Metadata: {metadata}")
            
            if kwargs:
                print(f"\n🛠️ 其他参数:")
                print(json.dumps(kwargs, indent=2, ensure_ascii=False, default=str))
            
            print("="*80)
    
    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """当LLM结束运行时调用"""
        if self.verbose:
            print("\n" + "="*80)
            print("✅ LLM 调用结束")
            print("="*80)
            
            print(f"\n📤 LLM 响应:")
            print(f"生成数量: {len(response.generations)}")
            
            for i, generation_list in enumerate(response.generations):
                print(f"\n--- 生成批次 {i+1} ---")
                for j, generation in enumerate(generation_list):
                    print(f"生成 {j+1}: {generation.text}")
                    if hasattr(generation, 'message') and generation.message:
                        print(f"消息类型: {type(generation.message).__name__}")
                        if hasattr(generation.message, 'tool_calls') and generation.message.tool_calls:
                            print(f"工具调用: {json.dumps(generation.message.tool_calls, indent=2, ensure_ascii=False)}")
            
            # 打印LLM输出的统计信息
            if response.llm_output:
                print(f"\n📊 LLM 输出统计:")
                print(json.dumps(response.llm_output, indent=2, ensure_ascii=False, default=str))
            
            print("="*80)

