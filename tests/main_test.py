import asyncio
import sys
from asyncio import WindowsSelectorEventLoopPolicy

import pytest
from loguru import logger

from main import stream_graph_updates
from thread_manager import ThreadManager
from tools.tool_manager import ToolManager

asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

logger.remove()  # 先移除默认的控制台输出
logger.add(sys.stdout, level="INFO")


@pytest.mark.asyncio
async def test_run():
    thread_manager = ThreadManager()
    thread_id = await thread_manager.create_thread()

    config = {"configurable": {"thread_id": thread_id, "project_id": "3545e9c5-5651-426d-a858-6ea4eed1f8a1"}}
    tool_manager = ToolManager(
        thread_manager, config["configurable"]["project_id"], config["configurable"]["thread_id"]
    )
    tool_manager.register_all_tools()

    # message = "回答问题前先仔细思考并创建相关的任务，再根据创建的任务，一步一步执行，获取最后的结果。现在，我的问题是: 今天福州和厦门的温度谁更高？"
    message = "今天福州和厦门的温度谁更高？"
    await thread_manager.add_message(
        thread_id=thread_id, type="user", content={"role": "user", "content": message}, is_llm_message=True
    )

    await stream_graph_updates(message, thread_manager, config)
