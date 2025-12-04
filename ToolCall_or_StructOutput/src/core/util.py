import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
load_dotenv()
# 从环境变量获取模型名称列表
model_names = os.getenv("MODEL_NAMES").split(",")

# API配置
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

def init_model(model_name: str, temperature: float = 0.1):
    """初始化语言模型

    参数:
        model_name: 模型名称
        temperature: 温度参数（控制随机性）

    返回:
        ChatOpenAI实例
    """
    return ChatOpenAI(
        model=model_name,
        openai_api_key=API_KEY,
        openai_api_base=BASE_URL,
        temperature=temperature
    )
