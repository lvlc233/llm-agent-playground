"""
Utility functions for thin_king project.

This module contains shared utility functions used across different experiment versions.
"""

import asyncio
from typing import Any, Dict, List, Optional
from langchain_core.messages import AnyMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool


class ThinkingUtils:
    """Utility class for thinking-related operations."""
    
    @staticmethod
    async def execute_tool_safely(tool: BaseTool, args: Dict[str, Any]) -> str:
        """
        Safely execute a tool with error handling.
        
        Args:
            tool: The tool to execute
            args: Arguments for the tool
            
        Returns:
            Tool execution result or error message
        """
        try:
            return await tool.ainvoke(args)
        except Exception as e:
            return f"Error executing tool {tool.name}: {str(e)}"
    
    @staticmethod
    def create_tool_dict(tools: List[BaseTool]) -> Dict[str, BaseTool]:
        """
        Create a mapping of tool names to tools for easy lookup.
        
        Args:
            tools: List of tools
            
        Returns:
            Dictionary mapping tool names to tool objects
        """
        return {tool.name: tool for tool in tools}
    
    @staticmethod
    def format_messages_for_display(messages: List[AnyMessage]) -> str:
        """
        Format messages for display/logging.
        
        Args:
            messages: List of messages
            
        Returns:
            Formatted string representation
        """
        formatted = []
        for msg in messages:
            if hasattr(msg, 'content'):
                formatted.append(f"{type(msg).__name__}: {msg.content}")
            else:
                formatted.append(f"{type(msg).__name__}: {str(msg)}")
        return "\n".join(formatted)


class GraphUtils:
    """Utility class for graph-related operations."""
    
    @staticmethod
    def should_continue_thinking(state: Any) -> bool:
        """
        Determine if the agent should continue thinking based on state.
        
        Args:
            state: Current state object
            
        Returns:
            True if should continue thinking, False otherwise
        """
        if not hasattr(state, 'messages') or not state.messages:
            return False
            
        last_message = state.messages[-1]
        
        # Continue if last message has tool calls
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return True
            
        # Continue if it's a thinking-related message
        if isinstance(last_message, AIMessage):
            content = getattr(last_message, 'content', '')
            thinking_keywords = ['思考', '想想', '考虑', '分析']
            return any(keyword in content for keyword in thinking_keywords)
            
        return False
    
    @staticmethod
    def create_system_message(prompt: str, context: Optional[Dict[str, Any]] = None) -> SystemMessage:
        """
        Create a system message with optional context injection.
        
        Args:
            prompt: Base prompt text
            context: Optional context to inject into prompt
            
        Returns:
            SystemMessage object
        """
        if context:
            # Simple context injection - can be enhanced as needed
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            full_prompt = f"{prompt}\n\n<context>\n{context_str}\n</context>"
        else:
            full_prompt = prompt
            
        return SystemMessage(content=full_prompt)


class ExperimentUtils:
    """Utility class for experiment management."""
    
    @staticmethod
    def get_experiment_config(version: str) -> Dict[str, Any]:
        """
        Get configuration for specific experiment version.
        
        Args:
            version: Experiment version (v1, v2, etc.)
            
        Returns:
            Configuration dictionary
        """
        configs = {
            "v1": {
                "recursion_limit": 100,
                "thinking_enabled": False,
                "tools_enabled": True,
            },
            "v2": {
                "recursion_limit": 100,
                "thinking_enabled": True,
                "meta_thinking": True,
                "tools_enabled": True,
            },
            "v3": {
                "recursion_limit": 100,
                "thinking_enabled": True,
                "tool_thinking": True,
                "tools_enabled": True,
            },
            "v4": {
                "recursion_limit": 100,
                "thinking_enabled": True,
                "batch_thinking": True,
                "batch_size": 5,
                "tools_enabled": True,
            },
            "v5": {
                "recursion_limit": 100,
                "thinking_enabled": True,
                "network_thinking": True,
                "memory_enabled": True,
                "tools_enabled": True,
            },
        }
        
        return configs.get(version, configs["v1"])
    
    @staticmethod
    def validate_experiment_version(version: str) -> bool:
        """
        Validate if experiment version is supported.
        
        Args:
            version: Version string to validate
            
        Returns:
            True if valid, False otherwise
        """
        valid_versions = ["v1", "v2", "v3", "v3.5", "v4", "v5"]
        return version in valid_versions