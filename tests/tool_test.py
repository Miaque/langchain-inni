import uuid

import pytest
from loguru import logger

from thread_manager import ThreadManager
from tools.task_list_tool import TaskListTool


@pytest.mark.asyncio
async def test_create_tasks():
    thread_manager = ThreadManager()

    tool = TaskListTool("demo", thread_manager, str(uuid.uuid4()))
    tool_result = await tool.create_tasks()
    logger.info(f"tool result: {tool_result}")
