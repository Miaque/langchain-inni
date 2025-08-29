from typing import Optional

from loguru import logger

from configs import app_config
from tools.base_tool import BaseTool
from tools.message_tool import MessageTool
from tools.task_list_tool import TaskListTool
from tools.tool_registry import ToolRegistry
from tools.web_search_tool import SandboxWebSearchTool


class ToolManager:
    def __init__(self, tool_registry: ToolRegistry, thread_id: str):
        self.thread_id = thread_id
        self.tool_registry = tool_registry

    def register_all_tools(self, disabled_tools: Optional[list[str]] = None):
        """Register all available tools by default, with optional exclusions.

        Args:
            disabled_tools: List of tool names to exclude from registration
        """
        disabled_tools = disabled_tools or []

        logger.debug(f"Registering tools with disabled list: {disabled_tools}")

        # Core tools - always enabled
        self._register_core_tools()

        # Sandbox tools
        # self._register_sandbox_tools(disabled_tools)

        # Data and utility tools
        # self._register_utility_tools(disabled_tools)

        # Browser tool
        # self._register_browser_tool(disabled_tools)

        logger.debug(f"Tool registration complete. Registered tools: {list(self.tool_registry.tools.keys())}")

    def add_tool(self, tool_class: type[BaseTool], function_names: Optional[list[str]] = None, **kwargs):
        """Add a tool to the ThreadManager."""
        self.tool_registry.register_tool(tool_class, function_names, **kwargs)

    def _register_core_tools(self):
        """Register core tools that are always available."""
        # self.thread_manager.add_tool(ExpandMessageTool, thread_id=self.thread_id, thread_manager=self.thread_manager)
        self.add_tool(MessageTool)
        self.add_tool(TaskListTool, thread_id=self.thread_id)

    def _register_sandbox_tools(self, disabled_tools: list[str]):
        """Register sandbox-related tools."""
        sandbox_tools = [
            ("sb_shell_tool", SandboxShellTool, {"project_id": self.project_id, "thread_manager": self.thread_manager}),
            ("sb_files_tool", SandboxFilesTool, {"project_id": self.project_id, "thread_manager": self.thread_manager}),
            (
                "sb_deploy_tool",
                SandboxDeployTool,
                {"project_id": self.project_id, "thread_manager": self.thread_manager},
            ),
            (
                "sb_expose_tool",
                SandboxExposeTool,
                {"project_id": self.project_id, "thread_manager": self.thread_manager},
            ),
            (
                "web_search_tool",
                SandboxWebSearchTool,
                {"project_id": self.project_id, "thread_manager": self.thread_manager},
            ),
            (
                "sb_vision_tool",
                SandboxVisionTool,
                {"project_id": self.project_id, "thread_id": self.thread_id, "thread_manager": self.thread_manager},
            ),
            (
                "sb_image_edit_tool",
                SandboxImageEditTool,
                {"project_id": self.project_id, "thread_id": self.thread_id, "thread_manager": self.thread_manager},
            ),
            (
                "sb_presentation_outline_tool",
                SandboxPresentationOutlineTool,
                {"project_id": self.project_id, "thread_manager": self.thread_manager},
            ),
            (
                "sb_presentation_tool",
                SandboxPresentationTool,
                {"project_id": self.project_id, "thread_manager": self.thread_manager},
            ),
            (
                "sb_sheets_tool",
                SandboxSheetsTool,
                {"project_id": self.project_id, "thread_manager": self.thread_manager},
            ),
            (
                "sb_web_dev_tool",
                SandboxWebDevTool,
                {"project_id": self.project_id, "thread_id": self.thread_id, "thread_manager": self.thread_manager},
            ),
            (
                "sb_upload_file_tool",
                SandboxUploadFileTool,
                {"project_id": self.project_id, "thread_manager": self.thread_manager},
            ),
        ]

        for tool_name, tool_class, kwargs in sandbox_tools:
            if tool_name not in disabled_tools:
                self.add_tool(tool_class, **kwargs)
                logger.debug(f"Registered {tool_name}")

    def _register_utility_tools(self, disabled_tools: list[str]):
        """Register utility and data provider tools."""
        if app_config.RAPID_API_KEY and "data_providers_tool" not in disabled_tools:
            self.add_tool(DataProvidersTool)
            logger.debug("Registered data_providers_tool")

    def _register_browser_tool(self, disabled_tools: list[str]):
        """Register browser tool."""
        if "browser_tool" not in disabled_tools:
            from agent.tools.browser_tool import BrowserTool

            self.add_tool(
                BrowserTool, thread_id=self.thread_id
            )
            logger.debug("Registered browser_tool")
