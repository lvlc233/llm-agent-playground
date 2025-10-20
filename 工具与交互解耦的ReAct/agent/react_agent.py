import asyncio
from enum import Enum
from typing import  Any
from langchain_core.messages import  HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langgraph.graph.message import MessagesState
from langgraph.graph import StateGraph
from langchain.agents import ToolNode
from datetime import datetime
# 加载环境变量
load_dotenv(dotenv_path=".env", override=True)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv('LANGSMITH_ENDPOINT')
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGSMITH_API_KEY')
os.environ["LANGCHAIN_PROJECT"] = os.getenv('LANGSMITH_PROJECT')


"""
工具
"""
# 工具注册表
@tool("搜索")
async def search_tool(query: str) -> str:
    """搜索工具 - 模拟网络搜索"""
    # 模拟异步搜索延迟
    await asyncio.sleep(2)
    
    # 模拟搜索结果
    results = [
        f"关于'{query}'的搜索结果1：这是一个相关的信息...",
        f"关于'{query}'的搜索结果2：这里有更多详细内容...",
        f"关于'{query}'的搜索结果3：相关的技术文档和资料..."
    ]
    
    return "\n".join(results)


@tool("获取时间")
def get_time_tool() -> str:
    """获取当前时间"""
    current_time = datetime.now()
    return f"当前时间：{current_time.strftime('%Y-%m-%d %H:%M:%S')}"

@tool("文件读取")   
async def read_file_tool(file_path: str) -> str:
    """文件读取工具"""
    try:
        # 模拟文件读取延迟
        await asyncio.sleep(1)
        
        # 这里应该实现真实的文件读取逻辑
        # 为了演示，返回模拟内容
        return f"文件 '{file_path}' 的内容：\n这是文件的模拟内容..."
    except Exception as e:
        return f"文件读取错误：{str(e)}"

#  这样子好麻烦啊,解耦...这就是MCP的好处了hhh
func_map=[search_tool,get_time_tool,read_file_tool]

class LLMState(Enum):
    RUNNING="RUNNING"
    INTERRUPT="INTERRUPT"
    WAITING="WAITING"
    DONE="DONE"


"""
状态
"""
class ReActState(MessagesState):
    tool_result:Any
    llm_state:LLMState

"""
初始化模型
"""
llm = ChatOpenAI(
        model_name=os.getenv('OPENAI_MODEL_NAME'),
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('OPENAI_BASE_URL'),
    )
# .bind_tools(func_map)
"""
节点
"""
# LLM
# 1,分析需求并给出答案
# 2,验证需求是否完成
    # 已经完成->给出答案
    # 尚未完成
        #分析未完成的原因(多选)
            #1,需要使用工具->调用工具
            #2,需要等待工具结果->等待
            #3,思考角度不够充分->思考
        # 执行对应的处理


# tool_node = ToolNode(
#     tools=[get_available_tools])

def astream_call_llm(state:ReActState):
    messages=state.get("messages")
    res=llm.invoke(messages)
    ai=AIMessage(content=res.content)
    return {"llm_state":LLMState}
    # messages=state.get("messages")
    # if not isinstance(messages[0],SystemMessage):
    #     messages.insert(0,SystemMessage(content=system_prompt_template))
    
    # cache=AIMessage(content="")
    # async for chunk in self.llm.astream(messages):
    #     cache.content += chunk.content or ""
    # return {"messages":cache}

def router(state:ReActState)->str:
    if isinstance(state.get("messages")[-1],ToolMessage):
        return "tool"
    else:
        return "llm"
    

"""
图构建
"""
graph=StateGraph(MessagesState)
graph.add_node("llm",astream_call_llm)
graph.set_entry_point("llm").set_finish_point("llm")
app=graph.compile()

"""
执行
"""
res=app.astream({"messages":[HumanMessage(content="使用200个字介绍python")]},stream_mode="messages")
n=0


async def main():
    async for chunk in res:
        print(chunk[0].content)

if __name__ == "__main__":
    asyncio.run(main())

# graph.add_node("llm",astream_call_llm)
# graph.add_node("tool",tool_node)
# graph.set_entry_point("llm")
# graph.add_conditional_edges("llm",
#     self.router,
#     {
#         "END":"END",
#         "tool":"tool"
#     })
# graph.add_edge("tool","llm")
# graph.compile()




    # async def stream_response(self, messages: List[BaseMessage]) -> AsyncGenerator[Any, None]:
    #     """流式响应生成"""
    #     # 构建完整的消息列表
    #     full_messages = [HumanMessage(content=self.system_prompt)] + messages
        
    #     async for chunk in self.llm.astream(full_messages):
    #         yield chunk
    
    # async def stream_with_context(self, messages: List[BaseMessage]) -> AsyncGenerator[Any, None]:
    #     """带上下文的流式响应"""
    #     async for chunk in self.stream_response(messages):
    #         yield chunk
    
   
    
    # async def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
    #     """执行工具"""
    #     if tool_name not in self.tools:
    #         return f"错误：工具 '{tool_name}' 不存在"
        
    #     try:
    #         tool_func = self.tools[tool_name]
            
    #         # 如果是异步工具
    #         if asyncio.iscoroutinefunction(tool_func):
    #             result = await tool_func(**tool_input)
    #         else:
    #             result = tool_func(**tool_input)
            
    #         return str(result)
    #     except Exception as e:
    #         return f"工具执行错误: {str(e)}"
    
    # def should_continue_conversation(self, content: str) -> bool:
    #     """判断是否应该继续对话（即使有工具调用）"""
    #     # 检查是否包含继续对话的指示
    #     continue_indicators = [
    #         "让我们继续", "我们可以继续", "同时", "另外", 
    #         "在工具执行期间", "工具正在执行", "我们先"
    #     ]
        
    #     return any(indicator in content for indicator in continue_indicators)
    
    # def extract_conversation_part(self, content: str) -> str:
    #     """提取对话部分（排除工具调用部分）"""
    #     # 移除工具调用相关的内容
    #     lines = content.split('\n')
    #     conversation_lines = []
        
    #     skip_next = False
    #     for line in lines:
    #         if '思考:' in line or '行动:' in line or '行动输入:' in line:
    #             skip_next = True
    #             continue
    #         if skip_next and line.strip() == '':
    #             skip_next = False
    #             continue
    #         if not skip_next:
    #             conversation_lines.append(line)
        
    #     return '\n'.join(conversation_lines).strip()
    
    # async def process_user_input(self, user_input: str, conversation_history: List[BaseMessage]) -> Dict[str, Any]:
        # """处理用户输入"""
        # # 添加用户消息到历史
        # messages = conversation_history + [HumanMessage(content=user_input)]
        
        # # 生成响应
        # full_response = ""
        # async for chunk in self.stream_response(messages):
        #     content = getattr(chunk, 'content', '')
        #     if content:
        #         full_response += content
        
        # # 解析工具调用
        # tool_calls = self.parse_tool_calls(full_response)
        
        # # 提取对话部分
        # conversation_part = self.extract_conversation_part(full_response)
        
        # return {
        #     "full_response": full_response,
        #     "conversation_part": conversation_part,
        #     "tool_calls": tool_calls,
        #     "should_continue": self.should_continue_conversation(full_response)
        # }