import os
from typing import List, Union, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def get_llm(model_name: str = None, temperature: float = 0.1):
    """
    Get a ChatOpenAI instance configured from environment variables.
    """
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    default_models = os.getenv("MODEL_NAMES", "").split(",")

    if not model_name and default_models:
        model_name = default_models[0]

    if not model_name:
        raise ValueError("No model name provided and MODEL_NAMES not set in .env")

    # Handle specific model quirks if necessary (e.g. some models don't support tool_choice="required")

    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=temperature,
    )
    return llm

def get_multiple_llms(model_names: Union[str, List[str]] = None, temperature: float = 0.1) -> List[ChatOpenAI]:
    """
    Get multiple ChatOpenAI instances for batch testing.

    Args:
        model_names: List of model names or comma-separated string. If None, uses all models from MODEL_NAMES.
        temperature: Temperature setting for all models.

    Returns:
        List of ChatOpenAI instances.
    """
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")

    if model_names is None:
        model_names = os.getenv("MODEL_NAMES", "").split(",")
    elif isinstance(model_names, str):
        model_names = [name.strip() for name in model_names.split(",")]

    if not model_names or model_names == [""]:
        raise ValueError("No model names provided and MODEL_NAMES not set in .env")

    llms = []
    for model_name in model_names:
        if model_name.strip():  # Skip empty strings
            llm = ChatOpenAI(
                model=model_name.strip(),
                openai_api_key=api_key,
                openai_api_base=base_url,
                temperature=temperature,
            )
            llms.append(llm)

    return llms

def get_available_models():
    """Get list of available models from environment."""
    models = os.getenv("MODEL_NAMES", "").split(",")
    return [model.strip() for model in models if model.strip()]
