
from pydantic import BaseModel, Field
from typing import Literal,Optional
from langchain_core.tools import BaseTool
from langchain_core.messages import AnyMessage,ToolMessage,SystemMessage,RemoveMessage,HumanMessage 
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

# 思考的类型
"""
联想：一个念头勾出另一个，像树枝不断分岔，不急着收敛。
演绎：手上有自认为可靠的“如果”，就把它推向“那么”，看会不会推出荒谬。
归纳：把散落的个例往一处拢，试着描一条轮廓线，心里却随时预备“下一只乌鸦可能不是黑的”。
类比：把A的骨架搬到B身上，借熟悉度照亮陌生域，但随时警惕“像”不等于“是”。
反事实：把时间线拨回分叉口，问“假如当时……”，用来探测因果哪一环最脆弱。
外部化：把念头甩到纸面、模型、对话里，让眼睛代替工作记忆，借外部符号继续推。
自反：把镜头对准正在拍的自己，问“我刚才那步为什么信？凭什么疑？”——这是唯一能对前面六条动刀的工序。
"""
meta_thinks=[
    {"Association":"one idea hooks another, branch after branch, no hurry to converge."},
    {"Deduction":"take a premise you trust and push it to its “then,” watching for absurdity."},
    {"Induction":"pile the scattered cases together, sketch a contour, ready for the next black swan."},
    {"Analogy":"lift the skeleton of A onto B, letting familiarity light up the strange—while remembering “similar” is not “same.”"},
    {"Counterfactual":"rewind the timeline, ask “if only…,” to probe which causal link snaps first."},
    {"Externalization":"off-load the thought onto paper, model, or conversation; let the eyes take over from working memory and push the symbol-loop further."},
    {"Reflection":"turn the camera on the camera-operator—ask “why did I just believe that? what lets me doubt it?” The only move that can edit the other six."},
]
# 提取 meta_thinks 的 keys 用于 Literal 类型
meta_think_types = tuple(list(think_dict.keys())[0] for think_dict in meta_thinks)
class State(BaseModel):
    """状态类,用于存储状态"""
    messages: Annotated[List[AnyMessage], add_messages] = []
    action: Optional[str] = None
    think_type: Optional[Literal[meta_think_types]] = None
# 这是一个枚举问题,llm中像是让LLM固定输出一个范围内的固定的若干个选项,即选择的最优解是什么呢?
# 回答:就是结构化输出 





class Command_t(BaseModel):
    command: Literal["think", "speak"] = Field(description="""if you want think output think, 
                                                            if you want speak output speak,
                                                            but the speak is your end select,
                                                            so,if you want think something,
                                                            you must think first""")

# 指令的目的是:识别thinks和speak,因此这是一个分类任务,分类任务的最佳做法是结构化输出,因此这里选择结构化输出
async def command_build(state: State):
     history=state.messages
     llm=get_model().with_structured_output(Command_t)
     response = await llm.ainvoke(history)
     state.action=response.command
     return state

# 具体的思考内容
class ThinkSelect(BaseModel):
    think_type: Literal[meta_think_types] = Field(description=f"""now, you see the history and want think,
                                                                 you had think type select from that,
                                                                {meta_thinks}
                                                                you must think first""")
async def think_select(state: State):
    """思考选择节点,用于选择思考的类型"""
    history=state.messages
    llm=get_model().with_structured_output(ThinkSelect)
    response = await llm.ainvoke(history)
    state.think_type=response.think_type
    return state

async def think_build(state: State):
    """思考节点,用于思考"""
    history=state.messages
    llm=get_model()
    response = await llm.ainvoke([*history,SystemMessage(content=f"you see the history and want think:{state.think_type}")])
    state.think_type=response.think_type
    return state
# async def llm_call(state: State):
#     """LLM 调用节点，保证系统提示始终位于 messages 最顶部"""
#     if state.messages:
#         # 1. 把系统提示抽出来（如果之前加过就跳过，这里直接覆盖）
    
#         sys_msg = SystemMessage(content=base_model_prompt)
#         # 2. 去掉旧系统消息（可选，防止重复）
#         other_msgs = [m for m in state.messages if not isinstance(m, SystemMessage)]
#         # 3. 重新拼装：系统提示在最前，其余保持原顺序
#         state.messages = [sys_msg, *other_msgs]

#     think_model = get_think_model(tool_kit)
#     response = await think_model.ainvoke(state.messages)
#     if isinstance(state.messages[-1],ToolMessage):
#         return {"messages": [response]}
#     return {"messages": [response]}

# async def tool_node(state:State):
#     """工具节点,用于执行工具"""
#     if (state.messages[-1].tool_calls):
#         tool_node = ThinkToolNode(tools=tool_kit)
#         tool_outputs = await tool_node.excute(state.messages[-1])
    
#         return Command(
#                     goto="llm_call",
#                     update={"messages":tool_outputs}
#                 )
#     return Command(
#                 goto=END,
#             )



graph_build = StateGraph(State)
graph_build.add_node("command",command_build)
graph_build.add_node("think_select",think_select)
graph_build.add_edge(START,"command")
graph_build.add_edge("command","think_select")



graphv2 = graph_build.compile(name="agent").with_config(
            config={
                "recursion_limit": 100
            }
        )
