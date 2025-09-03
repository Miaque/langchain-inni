import sys
import uuid

import pytest
from loguru import logger

from thread_manager import ThreadManager
from tools.sb_files_tool import SandboxFilesTool
from tools.task_list_tool import TaskListTool

logger.remove()
logger.add(sys.stderr, level="INFO")


@pytest.fixture(scope="module")
def manager():
    logger.info("初始化 thread manager ...")
    return ThreadManager()


@pytest.mark.asyncio
async def test_create_tasks():
    tool = TaskListTool("demo", manager, str(uuid.uuid4()))
    tool_result = await tool.create_tasks(
        [
            {
                "title": "天气查询任务",
                "tasks": ["确认当前日期", "获取福州实时天气", "获取厦门实时天气", "比较两地温度", "创建温度对比报告"],
            }
        ]
    )
    logger.info(f"tool result: {tool_result}")


@pytest.mark.asyncio
async def test_create_file():
    tool = SandboxFilesTool("34b90f9b-8601-4b29-bf75-497ed8fb924f", manager)
    result = await tool.create_file(file_path="test.txt", file_contents="Hello, world!")
    logger.info("tool result: {}", result)
