import inspect
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, TypedDict

import aiofiles
from langchain.agents import Tool
from langchain.tools import StructuredTool
from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.constants import START
from langgraph.graph import StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt
from loguru import logger

from configs import app_config
from tools.tool_manager import ToolManager
from tools.tool_registry import ToolRegistry
from utils.current_date import get_current_date_info

logger.remove()  # 先移除默认的控制台输出
logger.add(sys.stdout, level="INFO")

tool_registry = ToolRegistry()


def get_tools() -> list[Tool]:
    functions = tool_registry.get_available_functions()
    tools = []

    for name, func in functions.items():
        desc = tool_registry.get_tool(name)["schema"].schema["function"]["description"]
        params = tool_registry.get_tool(name)["schema"].schema["function"]["parameters"]
        # tool = Tool(name=name, func=func, description=desc, args_schema=params)

        # 创建 StructuredTool，支持异步函数
        if inspect.iscoroutinefunction(func):
            tool = StructuredTool.from_function(
                func=func,
                name=name,
                description=desc,
                args_schema=params,
                coroutine=func,  # 指定异步函数
            )
        else:
            tool = StructuredTool.from_function(func=func, name=name, description=desc, args_schema=params)

        tools.append(tool)

    return tools


@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]


# llm = ChatOpenAI(model="Qwen/Qwen3-14B", base_url=app_config.base_url, api_key=app_config.api_key)
# tool = TavilySearch(tavily_api_key=app_config.tavily_api_key, max_results=app_config.max_results)
# tools = [tool, human_assistance] + get_core_tools()


def get_llm():
    llm = ChatLiteLLM(
        model=app_config.model_name,
        api_base=app_config.base_url,
        api_key=app_config.api_key,
        custom_llm_provider="openai",
    )

    tools = get_tools()
    llm_with_tools = llm.bind_tools(tools)

    return llm_with_tools


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    llm = get_llm()
    return {"messages": [llm.invoke(state["messages"])]}


def get_graph(checkpointer) -> CompiledStateGraph:
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools=get_tools())
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )

    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    graph = graph_builder.compile(checkpointer=checkpointer)
    return graph


async def get_system_prompt() -> str:
    async with aiofiles.open("prompt.md", encoding="utf-8") as f:
        content = await f.read()

    return content


def setup_tools(tool_manager: ToolManager):
    tool_manager.register_all_tools()


async def stream_graph_updates(user_input: str, config):
    system_prompt = await get_system_prompt()
    system_prompt = system_prompt.replace("{{current_date}}", get_current_date_info())

    tool_manager = ToolManager(tool_registry=tool_registry, thread_id=config["configurable"]["thread_id"])
    setup_tools(tool_manager)

    async with AsyncPostgresSaver.from_conn_string(app_config.SQLALCHEMY_DATABASE_URI) as checkpointer:
        graph = get_graph(checkpointer)

        async for event in graph.astream(
            {"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]},
            config,
            stream_mode="values",
        ):
            if "messages" in event:
                logger.info(event["messages"][-1].pretty_print())


if __name__ == "__main__":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    async def run():
        user_input = "今天是几号，今天福州和厦门的温度谁更高"
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        await stream_graph_updates(user_input, config)

    async def get_history_state():
        config = {"configurable": {"thread_id": "1", "checkpoint_id": "1f084048-2825-6f93-8004-484a5e04f341"}}

        async with AsyncPostgresSaver.from_conn_string(app_config.SQLALCHEMY_DATABASE_URI) as checkpointer:
            graph = get_graph(checkpointer)

            state = graph.get_state(config)
            logger.info(state)

    async def arun():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, get_history_state)
            
    asyncio.run(get_history_state())
