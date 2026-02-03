"""
Configuration management for thin_king project.

This module provides unified configuration for all experiment versions.
"""

import os
import logging
from typing import List, Optional
from langchain.chat_models import init_chat_model
from langchain_core.tools import BaseTool
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path="src/.env", override=True)

# LangSmith configuration
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv('LANGSMITH_API_KEY', '')
os.environ["LANGSMITH_PROJECT"] = os.getenv('LANGSMITH_PROJECT', 'thin_king')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class ModelConfig:
    """Model configuration class."""
    
    def __init__(self):
        self.openai_model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
        self.think_model_name = os.getenv("THINK_MODEL_NAME_V4", "gpt-3.5-turbo")
    
    def get_base_model(self):
        """Get the base chat model."""
        return init_chat_model(
            model=f"openai:{self.openai_model_name}",
        )
    
    def get_think_model(self, tools: Optional[List[BaseTool]] = None):
        """Get the thinking model with optional tools binding."""
        chat_model = init_chat_model(
            model=f"openai:{self.think_model_name}",
        )
        
        if tools:
            return chat_model.bind_tools(tools)
        return chat_model
    
    def get_model_for_version(self, version: str):
        """Get model configuration for specific experiment version."""
        if version == "v3.5":
            return self.get_base_model()
        elif version == "v4":
            return self.get_think_model()
        else:
            return self.get_base_model()


# Global configuration instance
config = ModelConfig()

# Convenience functions for backward compatibility
def get_model():
    """Get base model - for backward compatibility."""
    return config.get_base_model()

def get_model_think_v3_5():
    """Get model for v3.5 - for backward compatibility."""
    return config.get_model_for_version("v3.5")

def get_think_model(tools: List[BaseTool]):
    """Get think model with tools - for backward compatibility."""
    return config.get_think_model(tools)

def get_model_think_v4():
    """Get model for v4 - for backward compatibility."""
    return config.get_model_for_version("v4")