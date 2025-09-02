from typing import Optional

from tools.base_tool import BaseTool
from tools.tool_registry import ToolRegistry


class ThreadManager:
    def __init__(self):
        self.tool_registry = ToolRegistry()

    def add_tool(self, tool_class: type[BaseTool], function_names: Optional[list[str]] = None, **kwargs):
        """Add a tool to the ThreadManager."""
        self.tool_registry.register_tool(tool_class, function_names, **kwargs)
