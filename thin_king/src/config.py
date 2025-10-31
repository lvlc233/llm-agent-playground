from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv(dotenv_path="src\.env",override=True)
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv('LANGSMITH_API_KEY')
os.environ["LANGSMITH_PROJECT"] = os.getenv('LANGSMITH_PROJECT')

from langchain_core.tools import BaseTool
from typing import List



def get_model():
    return init_chat_model(
        model="openai:"+os.getenv("OPENAI_MODEL_NAME"),
    ) 
def get_model_think_v3_5():
    return init_chat_model(
        model="openai:"+os.getenv("OPENAI_MODEL_NAME"),
    ) 

def get_think_model(tools: List[BaseTool]):
    chat_model = init_chat_model(
        model="openai:"+os.getenv("THINK_MODEL_NAME_V4"),
        
    ) 

    return chat_model.bind_tools(tools)
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日志格式
    handlers=[
        logging.FileHandler('app.log'),  # 输出到文件
        logging.StreamHandler()          # 输出到控制台
    ]
)