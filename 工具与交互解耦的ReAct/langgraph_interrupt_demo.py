from typing import Annotated, Any, List,Optional,Callable
from pydantic import BaseModel,Field
import asyncio
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage,BaseMessage,RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from dotenv import load_dotenv
from langgraph.types import Command,interrupt
from langchain_core.tools import tool
from langchain.agents import ToolNode

from enum import Enum

"""
初始化LLM
"""
# 加载环境变量
load_dotenv()
import os
model_name=os.getenv('OPENAI_MODEL_NAME')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv('LANGSMITH_ENDPOINT')
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGSMITH_API_KEY')
os.environ["LANGCHAIN_PROJECT"] = os.getenv('LANGSMITH_PROJECT')

"""
工具定义
工具执行完成之后发送信息保存起来,等待llm节点处理
"""

# class AsyncToolNode():

#     async def arun(self, tool_call_id: str, inputs: str) -> str:
#         tool = self.tools[tool_call_id]
        # return await tool.arun(inputs)
config = {
    "configurable": {
        "thread_id": "1"}
}

# 
zhon=[]
#工具
@tool("网络搜索")
async def web_search(query: str,config:RunnableConfig)->Command:
    """用于搜索互联网"""
    # await asyncio.sleep(3)
    res= f"网络搜索结果: {query}..."
    """
    这里这么写是因为Command有一个坑点,或者说两个
    1,想要在工具函数中通过command来更新状态,那就必须使用ToolNode--->ToolNode内部有Command的处理--->除非自定义工具处理函数节点也实现了对Command的处理
    2,使用ToolNode更新的时候,只能是带有ToolMessages的messages
    3,所以写到最后,就兼容一下吧,
    """
    command= Command(
        update={"messages": [ToolMessage(content=res,tool_call_id=config.get("configurable",{}).get("tool_id"))],"done_task": [ToolMessage(content=res,tool_call_id=config.get("configurable",{}).get("tool_id"))]},
    )
    zhon.append([ToolMessage(content=res,tool_call_id=config.get("configurable",{}).get("tool_id"))])

    return command
#工具
@tool("本地搜索")
async def local_search(query: str,config:RunnableConfig)->Command:
    """用于搜索本地文件"""
    await asyncio.sleep(5)
    res= f"本地搜索结果: {query}..."
    command= Command(
        update={"messages": [ToolMessage(content=res,tool_call_id=config.get("configurable",{}).get("tool_id"))]}
        # ,
    )
    return command
tools=[web_search,local_search]
llm = ChatOpenAI(model=model_name, temperature=0.7).bind_tools(tools)

"""
状态定义
"""
class TaskEnum(Enum):
    RUNNING=0
    DONE=1
#  处理done_task,若消息中包含removeMessage(removeMessages的id为ToolID)

class ReActState(BaseModel):
    "存储LLM用的消息"
    messages:Annotated[List[BaseMessage],add_messages]=Field(default_factory=list)
    

    """
        多个task提交的合并处理 use ToolMessages
        以及
        llm节点消化移除的task处理 use RemoveMessages
    """
    @staticmethod
    def done_task_reduce(left: List[ToolMessage], right: List[BaseMessage]) -> List[ToolMessage]:
        remove_ids = {msg.id for msg in right if isinstance(msg, RemoveMessage)}
        filtered_left = [msg for msg in left if msg.tool_call_id not in remove_ids]
        filtered_right = [msg for msg in right if not isinstance(msg, RemoveMessage)]
        return filtered_left + filtered_right
    """
    提交任务队列
    """
    submit_task: List[AIMessage] = Field(default_factory=list)
    """
    已经完成的消息队列
    """
    done_task: Annotated[List[BaseMessage], done_task_reduce] = Field(default_factory=list)
    """
    是否等待
    """

"""
节点定义
"""
"""
任务提交节点:
    获取消息中的Tool消息
    将tool消息提交到提交到提交队列中
    告知llm任务已经提交
"""
class SubmitState(BaseModel):
    submit_task: List[AIMessage] = Field(default_factory=list)
    messages: List[BaseMessage] = Field(default_factory=list)
async def submit_task_node(state: SubmitState)->SubmitState:
    print("到达任务提交节点")
    """
    获取工具信息
    """
    tool_message = state.messages[-1]
    submit_task_list:List[AIMessage]=state.submit_task
    submit_task_list.append(tool_message)
    tool_node=ToolNode(tools=tools)
    """
    使用异步的方法来执行工具调用,并以不阻塞的方式做状态的提交
    """
    for task in submit_task_list:
        tool_calls = task.additional_kwargs.get("tool_calls")
        if tool_calls and len(tool_calls) > 0:
            tool_id = tool_calls[0].get("id")
            asyncio.create_task(tool_node.ainvoke(input={"messages":[task]},config={"configurable":{"tool_id":tool_id}}))
        else:
            print(f"Warning: Task {task} has no tool_calls in additional_kwargs")
    return {"submit_task":submit_task_list,
            "messages":[SystemMessage(content="任务已经提交,请思考是继续回复还是等待工具完成")]}


class InterruptibleLLMNode(BaseModel):
    current_chunk : Any = None
    tool_callback : Optional[Callable]=None
    messages_callback : Optional[Callable]=None
    async def astream_call_llm(self, state: ReActState):
        """初始化信息"""
        messages = state.messages
        content = ""
        tool_calls={
            "id":None,
            "name":None,
            "arguments":""
        }
        print("\n[开始] LLM开始生成回答...")
        try:
        # 流式调用LLM
            async for chunk in llm.astream(messages):
            # 检查中断标志
            # 防止LLM输出工具信息的时候被打断,保证工具调用的完整性
                if state.done_task and tool_calls.get("id") is not None:
                    """
                    当异步任务使用Command标记任务已经完成时候,打断模型输出
                    """
                    print("\n[中断] LLM输出已停止")
                    break
                """
                合并工具调用信息
                    判断chunk中是否有tool_calls
                    再检查
                    chunk中的additional_kwargs的tool_calls是否有具体的值
                    在chunk中tool的具体消息在chunk.additional_kwargs.get("tool_calls")之中
                    其中
                    第一条消息为要调用的工具函数
                    之后的消息为工具调用的参数
                    工具调用: [
                        {'index': 0, 'id': '0199476c9ab17dfc3c35daeefd158096', 'function': {'arguments': '', 'name': '网络搜索'}, 
                        'type': 'function'}]
                    工具调用: [{'index': 0, 'id': None, 'function': {'arguments': ' {"', 'name': None}, 'type': None}]
                """
                self.current_chunk=chunk
                if self._is_use_tool():
                    self._reduce_tool_chunk(tool_calls)
                    if self.tool_callback:
                        self.tool_callback(tool_calls)
                else:
                # 这是普通文本消息
                    # 累积内容
                    chunk_content = chunk.content or ""
                    content += chunk_content
                    if self.messages_callback:
                        self.messages_callback(chunk)
              
            """
            消息生成完毕
            或
            被打断时处理信息
                """
            #若已存在done_task,则说明工具调用完成
            if state.done_task:
                print("\n\n工具执行完成")
                # 封装工具信息并移除done_task中的任务
                return {"messages": state.messages.append(AIMessage(content=chunk_content))+[SystemMessage(content="在你思考过程中,你发现工具执行完成")], 
                        "submit_task": [],
                        # "done_task": [RemoveMessage(task.tool_call_id)for task in state.done_task],
                        }
            else:
                # 如果还有运行任务,等待任务完成
                # 这里其实不一定要等待工具调用完成,实际上可以是由LLM来判断是否要继续思考还是结束
                # 或者对接用户
                # 又或者单独封装一个判断是否等待
                # ...但是为了简单示范,就采用当llm不思考,并存在工具时等待工具的处理方式好了
                while state.submit_task:
                    if zhon:
                        print(zhon[0])
                        state.messages.append(AIMessage(content=chunk_content))
                        state.messages.append(SystemMessage(content="在你思考过程中,你发现工具执行完成"))
                        state.messages.append(zhon[0][0])
                        return {"messages": state.messages,
                                # 移除submit_task-->这里有点懒,就全清除了
                                "submit_task": [],
                                # "done_task": [RemoveMessage(task.tool_call_id)for task in state.done_task],
                                }
                #判断是工具调用还是结果输出
                if content == "":
                    additional_kwargs=self._build_additional_kwargs(tool_calls)
                return {"llm_interrupt_reason": None,"messages": [AIMessage(content=content,additional_kwargs=additional_kwargs)]}
        except Exception as e:
            import traceback
            import sys
            print("\n================== 详细错误信息 ==================")
            traceback.print_exc()

            # 2. 如果你想把堆栈也拿到字符串里（方便写日志或返回前端）
            exc_type, exc_value, exc_tb = sys.exc_info()
            full_trace = traceback.format_exception(exc_type, exc_value, exc_tb)
            full_trace_str = "".join(full_trace)
            # 3. 打印/记录
            print(full_trace_str)          # 控制台
            # logger.error(full_trace_str) # 如果用了 logging

            # 4. 返回给用户的内容（可选）
            content = f"LLM调用出错: {e}\n详情见服务端日志。"
            return {"llm_interrupt_reason": None,"messages": [AIMessage(content=content)]}
                # 检查是否在开始前就被中断
    def _is_use_tool(self)->bool:
        return hasattr(self.current_chunk, 'tool_calls') and self.current_chunk.additional_kwargs.get("tool_calls")
    def _reduce_tool_chunk(self,tool_calls:dict)->dict:
        # 这是工具调用消息
        chunk_tool_calls=self.current_chunk.additional_kwargs.get("tool_calls")[0]
        #函数消息获取
        if chunk_tool_calls.get("type")=="function":
            tool_calls["id"]=chunk_tool_calls.get("id")
            tool_calls["name"]=chunk_tool_calls.get("function").get("name")
        else:
            tool_calls["arguments"]+=chunk_tool_calls.get("function").get("arguments")
        return tool_calls
    def _build_additional_kwargs(self,tool_calls:dict)->dict:
        if tool_calls.get("id"):
            return {
                "tool_calls": [
                    {
                        "id": tool_calls.get("id"),
                        "type": "function",
                        "function": {
                            "name": tool_calls.get("name"),
                            "arguments": tool_calls.get("arguments")
                        },
                    },
                ]
            }
        else:
            return {}
"""
提交路由:
    1. 检查是否有工具调用
    2. 有则提交任务
    3. 无则结束
"""
# 这里为了简单,如果不用工具的话,就让它结束,但是这里可以加上一个反思的节点,或者使用结构化输出的方式,让模型生成带有think_end的字段来做判断
def from_llm_to_submit_task_router(state: ReActState)->str:
    
    
    if hasattr(state.messages[-1],'tool_calls'):
        print("==============="+state.messages[-1].content)
        return "submit_task_node"
    if isinstance(state.messages[-1],ToolMessage):
        return "llm"
    else:
        return END

# # 用来等待任务的节点
# # 由于在node中的State是
# def waitting_node

class InterruptDemo:
    def __init__(self,tool_callback:callable=None,messages_callback:callable=None):
        self.llm_node = InterruptibleLLMNode(tool_callback=tool_callback,messages_callback=messages_callback)
        self.graph = self._build_graph()
        self.running = False
    
    def _build_graph(self):
        """构建LangGraph"""
        # 整体的图结构和最基础的react是一样的,最核心内容在于llm异步流式输出和使用事件驱动的输出中断以及再输出过程
        graph = StateGraph(ReActState)
        graph.add_node("llm", self.llm_node.astream_call_llm)
        graph.add_node("submit_task_node", submit_task_node)
        graph.add_edge(START, "llm")
        graph.add_conditional_edges("llm", from_llm_to_submit_task_router,
            {
                "submit_task_node": "submit_task_node",
                "llm": "llm",
                END: END
            }
        )
        graph.add_edge("submit_task_node", "llm")
        return graph.compile()
    


async def main():
    await InterruptDemo(messages_callback=lambda x:print(x.content, end="", flush=True)).graph.ainvoke({
    "messages": [HumanMessage(content="搜索下北京天气")],   
},config=config)

if __name__ == "__main__":
    asyncio.run(main())
