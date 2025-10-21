from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage,SystemMessage,HumanMessage,AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph ,END
from dotenv import load_dotenv
from langchain_core.tools import tool
import os


load_dotenv()
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv('LANGSMITH_API_KEY')
os.environ["LANGSMITH_PROJECT"] = os.getenv('LANGSMITH_PROJECT')

llm=init_chat_model(
        model="openai:"+os.getenv("OPENAI_MODEL_NAME"),
    )


# 标注:为了排除干扰,即防止模型可能存在的拟合现象,例如add_tool,think_tool等这种一定程度上可以反应工具作用的函数,
# 测试时,我们使用asfhasflas的无意义函数名

def standard_tool_call_test():
    @tool
    def asfhasflas(a:str)->int:
        """
         在你需要思考时，调用asfhasflas函数,这个函数会帮助你进行思考。
        """
        
        return a

    system_prompt = """
       .
    """
    llm.bind_tools([asfhasflas])
    res=llm.invoke([SystemMessage(content=system_prompt),HumanMessage(content="思考这个问题`关于记忆的在LLM的思想`")])
    print(f"内容: {res.content}")
    print(f"是否有tool_calls: {hasattr(res, 'tool_calls') and res.tool_calls}")


# 等幂性测试01
# 做法 将标准工具调用的工具描述和系统提示词交换位置
# 预期结果: 其结果和标准工具调用的结果相似
def idempotent_tool_call_test01():
    @tool
    def asfhasflas(a:str)->int:
        """
        .
        """
        
        return a

    system_prompt = """
       在你需要思考时，调用asfhasflas函数,这个函数会帮助你进行思考。
    """
    llm.bind_tools([asfhasflas])
    res=llm.invoke([SystemMessage(content=system_prompt),HumanMessage(content="思考这个问题`关于记忆的在LLM的思想`")])
    print(f"内容: {res.content}")
    print(f"是否有tool_calls: {hasattr(res, 'tool_calls') and res.tool_calls}")


# 等幂性测试02
# 做法 在工具中写一般写在系统提示词的描述
# 预期结果: llm知道自己叫ASF
# 实验失败
# 应该会有影响的,因为think就被影响到了?吗?
def idempotent_tool_call_test02():
    @tool
    def asfhasflas(a:str)->int:
        """
        "messages":{你是一个助手,你的名字叫ASF,在别人问起你的时候你会告诉他们你叫ASF}
        """
        
        return a

    system_prompt = """
        .
    """
    llm.bind_tools([asfhasflas])
    res=llm.invoke([SystemMessage(content=system_prompt),HumanMessage(content="你叫什么?")])
    print(f"内容: {res.content}")
    print(f"是否有tool_calls: {hasattr(res, 'tool_calls') and res.tool_calls}")

# 拟态工具调用测试
# 不使用工具调用,仅使用系统提示词,系统提示词包括工具描述
# 预期结果: llm会根据系统提示词,进行工具调用
# 结果:输出了工具调用,但是是在content中
def mimicry_tool_call_test():
    system_prompt = """
         "tools": {
            "type": "function",
            "function": {
                "name": "asfhasflas",
                "description": "思考问题的时候调用这个工具,他会帮助你进行思考",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "str",
                            "description": "思考的内容"
                        },

                    },
                    "required": ["a"]
                }
            }
        }]
    """

    res=llm.invoke([SystemMessage(content=system_prompt),HumanMessage(content="思考这个问题`关于记忆的在LLM的思想`")])
    print(f"内容: {res.content}")
    print(f"是否有tool_calls: {hasattr(res, 'tool_calls') and res.tool_calls}")