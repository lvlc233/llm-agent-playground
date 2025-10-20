import asyncio
from typing import Dict, Optional, Callable, Any, List
from dataclasses import dataclass
from enum import Enum
import uuid
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class StreamStatus(Enum):
    IDLE = "idle"
    STREAMING = "streaming"
    INTERRUPTED = "interrupted"
    TOOL_EXECUTING = "tool_executing"
    RESUMING = "resuming"

@dataclass
class StreamSession:
    session_id: str
    status: StreamStatus
    current_task: Optional[asyncio.Task] = None
    accumulated_content: str = ""
    messages_history: List[BaseMessage] = None
    tool_results: Dict[str, Any] = None
    user_interrupt_callback: Optional[Callable] = None
    
    def __post_init__(self):
        if self.messages_history is None:
            self.messages_history = []
        if self.tool_results is None:
            self.tool_results = {}

class AsyncInterruptManager:
    """异步中断恢复管理器 - 支持流式输出的中断和恢复"""
    
    def __init__(self):
        self.sessions: Dict[str, StreamSession] = {}
        self.tool_execution_tasks: Dict[str, asyncio.Task] = {}
        
    def create_session(self, session_id: Optional[str] = None) -> str:
        """创建新的流式会话"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = StreamSession(
            session_id=session_id,
            status=StreamStatus.IDLE
        )
        return session_id
    
    def get_session(self, session_id: str) -> Optional[StreamSession]:
        """获取会话"""
        return self.sessions.get(session_id)
    
    async def start_streaming(self, session_id: str, stream_func: Callable, *args, **kwargs) -> bool:
        """开始流式输出"""
        session = self.get_session(session_id)
        if not session or session.status == StreamStatus.STREAMING:
            return False
        
        session.status = StreamStatus.STREAMING
        session.accumulated_content = ""
        
        # 创建流式任务
        session.current_task = asyncio.create_task(
            self._stream_with_interrupt_support(session, stream_func, *args, **kwargs)
        )
        
        return True
    
    async def _stream_with_interrupt_support(self, session: StreamSession, stream_func: Callable, *args, **kwargs):
        """支持中断的流式输出"""
        try:
            async for chunk in stream_func(*args, **kwargs):
                if session.status == StreamStatus.INTERRUPTED:
                    break
                
                # 累积内容
                content = getattr(chunk, 'content', str(chunk))
                if content:
                    session.accumulated_content += content
                    
                # 检查是否有工具调用
                if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                    await self._handle_tool_calls(session, chunk.tool_calls)
                    
                # 输出到控制台
                print(content, end="", flush=True)
                
                # 短暂让出控制权
                await asyncio.sleep(0.001)
                
        except asyncio.CancelledError:
            session.status = StreamStatus.INTERRUPTED
            print("\n[流式输出被中断]")
        except Exception as e:
            print(f"\n[流式输出错误: {e}]")
            session.status = StreamStatus.IDLE
        finally:
            if session.status == StreamStatus.STREAMING:
                session.status = StreamStatus.IDLE
    
    async def _handle_tool_calls(self, session: StreamSession, tool_calls: List[Any]):
        """处理工具调用 - 异步执行，不阻塞流式输出"""
        session.status = StreamStatus.TOOL_EXECUTING
        print("\n[检测到工具调用，异步执行中...]")
        
        for tool_call in tool_calls:
            tool_id = getattr(tool_call, 'id', str(uuid.uuid4()))
            
            # 异步执行工具
            task = asyncio.create_task(
                self._execute_tool_async(session, tool_call, tool_id)
            )
            self.tool_execution_tasks[tool_id] = task
    
    async def _execute_tool_async(self, session: StreamSession, tool_call: Any, tool_id: str):
        """异步执行工具"""
        try:
            # 模拟工具执行时间
            await asyncio.sleep(2)
            
            # 模拟工具结果
            tool_name = getattr(tool_call, 'function', {}).get('name', 'unknown_tool')
            result = f"工具 {tool_name} 执行完成，结果: 模拟数据"
            
            session.tool_results[tool_id] = result
            
            # 工具完成后，中断当前流式输出并恢复
            await self.interrupt_and_resume_with_tool_result(session.session_id, tool_id, result)
            
        except Exception as e:
            print(f"\n[工具执行错误: {e}]")
        finally:
            if tool_id in self.tool_execution_tasks:
                del self.tool_execution_tasks[tool_id]
    
    async def interrupt_stream(self, session_id: str) -> bool:
        """中断流式输出"""
        session = self.get_session(session_id)
        if not session or session.status != StreamStatus.STREAMING:
            return False
        
        session.status = StreamStatus.INTERRUPTED
        
        if session.current_task and not session.current_task.done():
            session.current_task.cancel()
            try:
                await session.current_task
            except asyncio.CancelledError:
                pass
        
        return True
    
    async def resume_stream(self, session_id: str, additional_context: str = "") -> bool:
        """恢复流式输出"""
        session = self.get_session(session_id)
        if not session or session.status not in [StreamStatus.INTERRUPTED, StreamStatus.TOOL_EXECUTING]:
            return False
        
        # 构造恢复上下文
        resume_context = self._build_resume_context(session, additional_context)
        
        # 重新开始流式输出
        from .react_agent import ReactAgent  # 避免循环导入
        agent = ReactAgent()
        
        return await self.start_streaming(
            session_id, 
            agent.stream_with_context, 
            resume_context
        )
    
    async def interrupt_and_resume_with_tool_result(self, session_id: str, tool_id: str, tool_result: str):
        """中断当前输出并使用工具结果恢复"""
        session = self.get_session(session_id)
        if not session:
            return
        
        # 中断当前流
        await self.interrupt_stream(session_id)
        
        print(f"\n[工具 {tool_id} 完成，结果已获取，恢复输出...]")
        
        # 使用工具结果恢复
        tool_context = f"\n\n工具执行结果: {tool_result}\n\n基于以上工具结果，请继续回答:"
        await self.resume_stream(session_id, tool_context)
    
    def _build_resume_context(self, session: StreamSession, additional_context: str = "") -> List[BaseMessage]:
        """构建恢复上下文"""
        if not session.messages_history:
            return [HumanMessage(content="请继续" + additional_context)]
        
        # 获取最后的用户消息
        last_human_msg = None
        for msg in reversed(session.messages_history):
            if isinstance(msg, HumanMessage):
                last_human_msg = msg
                break
        
        if not last_human_msg:
            return [HumanMessage(content="请继续" + additional_context)]
        
        # 构造恢复消息
        resume_content = f"""
原始问题: {last_human_msg.content}

已生成内容: {session.accumulated_content}

{additional_context}

请从中断处继续完成回答。
        """.strip()
        
        return [HumanMessage(content=resume_content)]
    
    def add_message_to_history(self, session_id: str, message: BaseMessage):
        """添加消息到历史记录"""
        session = self.get_session(session_id)
        if session:
            session.messages_history.append(message)
    
    def get_session_status(self, session_id: str) -> Optional[StreamStatus]:
        """获取会话状态"""
        session = self.get_session(session_id)
        return session.status if session else None
    
    def cleanup_session(self, session_id: str):
        """清理会话"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if session.current_task and not session.current_task.done():
                session.current_task.cancel()
            del self.sessions[session_id]