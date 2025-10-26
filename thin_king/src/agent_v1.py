
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.messages import AnyMessage,ToolMessage,SystemMessage,RemoveMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph ,END
from langgraph.types import Command
import asyncio
from typing import List,Annotated, TypedDict
from src.config import get_think_model,get_model
from src.prompt import tool_kit,base_model_prompt,curiosity_prompt,feel_prompt
class ThinkToolNode(BaseModel):
    tools: List[BaseTool] = []
    # Tool Execution Helper Function
    async def execute_tool_safely(self, tool, args):
        """Safely execute a tool with error handling."""
        try:
            return await tool.ainvoke(args)
        except Exception as e:
            return f"Error executing tool: {str(e)}"
    def _get_tool_dict(self):
        # 创建工具名和工具的映射,方便后续tool[name]调度工具
        tools_by_name = {
            tool.name: tool 
            for tool in self.tools
        }
        return tools_by_name
    async def excute(self,message:AnyMessage) -> ToolMessage:
        """执行工具,返回工具消息"""
    

        tool_calls_in_messages=message.tool_calls
        tools = self._get_tool_dict()
        # 创建任务
        tool_execution_tasks = [
            # 遍历获取消息中的工具调用,并执行
            self.execute_tool_safely(tools[tool_call["name"]], tool_call["args"]) 
            for tool_call in tool_calls_in_messages
        ]
        # 并发执行任务,获取结果
        observations = await asyncio.gather(*tool_execution_tasks)
        tool_outputs = [
            ToolMessage(
                content=observation,
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            ) 
            for observation, tool_call in zip(observations, tool_calls_in_messages)
        ]
        return tool_outputs


class State(BaseModel):
    """状态类,用于存储状态"""
    messages: Annotated[List[AnyMessage], add_messages] = []
    # tool_call_id: str = ""
    # history: Annotated[List[AnyMessage], add_messages] = []


async def llm_call(state: State):
    """LLM 调用节点，保证系统提示始终位于 messages 最顶部"""
    if state.messages:
        # 1. 把系统提示抽出来（如果之前加过就跳过，这里直接覆盖）
    
        sys_msg = SystemMessage(content=base_model_prompt)
        # 2. 去掉旧系统消息（可选，防止重复）
        other_msgs = [m for m in state.messages if not isinstance(m, SystemMessage)]
        # 3. 重新拼装：系统提示在最前，其余保持原顺序
        state.messages = [sys_msg, *other_msgs]

    think_model = get_think_model(tool_kit)
    response = await think_model.ainvoke(state.messages)
    if isinstance(state.messages[-1],ToolMessage):
        return {"messages": [response]}
    return {"messages": [response]}

async def tool_node(state:State):
    """工具节点,用于执行工具"""
    if (state.messages[-1].tool_calls):
        tool_node = ThinkToolNode(tools=tool_kit)
        tool_outputs = await tool_node.excute(state.messages[-1])
    
        return Command(
                    goto="llm_call",
                    update={"messages":tool_outputs}
                )
    return Command(
                goto=END,
            )



graph_build = StateGraph(State)
graph_build.add_node("llm_call",llm_call)
graph_build.add_node("tool_node",tool_node)
graph_build.add_edge(START,"llm_call")
graph_build.add_edge("llm_call","tool_node")


graphv1 = graph_build.compile(name="agent").with_config(
            config={
                "recursion_limit": 100
            }
        )

