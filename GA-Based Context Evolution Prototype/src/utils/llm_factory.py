import os
from typing import Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMFactory:
    """用于创建配置为SiliconFlow的ChatOpenAI实例的工厂。"""
    
    @staticmethod
    def create(model_name: str, temperature: float = 0.7) -> ChatOpenAI:
        api_key = os.getenv("SILICONFLOW_API_KEY")
        base_url = os.getenv("SILICONFLOW_BASE_URL")
        
        if not api_key:
            raise ValueError("未设置 SILICONFLOW_API_KEY 环境变量")
            
        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=temperature
        )
