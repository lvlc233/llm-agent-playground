import asyncio
from typing import Dict
from langchain_core.tools import tool,BaseTool
import json
import requests
from datetime import datetime



def get_available_tools():
    return func_map