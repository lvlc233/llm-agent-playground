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

def get_think_model(tools: List[BaseTool]):
    chat_model = init_chat_model(
        model="openai:"+os.getenv("OPENAI_MODEL_NAME"),
        
    ) 

    return chat_model.bind_tools(tools)
 