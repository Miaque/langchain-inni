import uuid

import pytest
from loguru import logger

from thread_manager import ThreadManager
from tools.task_list_tool import TaskListTool


@pytest.mark.asyncio
async def test_create_tasks():
    thread_manager = ThreadManager()

    tool = TaskListTool("demo", thread_manager, str(uuid.uuid4()))
    tool_result = await tool.create_tasks([{'title': '天气查询任务', 'tasks': ['确认当前日期', '获取福州实时天气', '获取厦门实时天气', '比较两地温度', '创建温度对比报告']}])
    logger.info(f"tool result: {tool_result}")
