import inspect
import json
from datetime import datetime
from typing import Annotated, Literal, TypedDict

import aiofiles
from langchain.tools import StructuredTool
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.constants import START
from langgraph.graph import StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import tools_condition
from langgraph.types import interrupt
from loguru import logger

from configs import WORK_DIR, app_config
from llm import get_reason_llm
from thread_manager import ThreadManager


class State(TypedDict):
    messages: Annotated[list, add_messages]
    native_max_auto_continues: int
    max_iterations: int
    enable_thinking: bool
    reasoning_effort: Literal["low", "medium", "high"]
    llm_max_tokens: int


class ToolManagerNode:
    def __init__(self, thread_manager: ThreadManager):
        self.thread_manager = thread_manager

    def __call__(self, state: State):
        functions = self.thread_manager.tool_registry.get_available_functions()
        tools = []

        for name, func in functions.items():
            desc = self.thread_manager.tool_registry.get_tool(name)["schema"].schema["function"]["description"]
            params = self.thread_manager.tool_registry.get_tool(name)["schema"].schema["function"]["parameters"]
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


class CoreAgent:
    def __init__(self, thread_manager: ThreadManager):
        self.thread_manager = thread_manager

    async def __call__(self, state: State, config: RunnableConfig):
        llm = get_reason_llm()
        message = llm.invoke(state["messages"])
        response_generator = self.thread_manager.response_processor.process_non_streaming_response(
            message, config["configurable"]["thread_id"]
        )

        if hasattr(response_generator, "__aiter__"):
            async for chunk in response_generator:
                logger.info("========: {}", chunk)
        else:
            logger.info("========: {}", response_generator)

        return {"messages": [message]}


def get_graph(thread_manager: ThreadManager, checkpointer=None) -> CompiledStateGraph:
    graph_builder = StateGraph(State)
    chatbot = CoreAgent(thread_manager=thread_manager)
    graph_builder.add_node("chatbot", chatbot)

    tool_manager_node = ToolManagerNode(thread_manager)
    graph_builder.add_node("tools", tool_manager_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )

    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    graph = graph_builder.compile(checkpointer=checkpointer) if checkpointer else graph_builder.compile()
    return graph


async def get_system_prompt(thread_manager: ThreadManager) -> str:
    async with aiofiles.open(WORK_DIR / "prompt.md", encoding="utf-8") as f:
        content = await f.read()

    now = datetime.now()
    datetime_info = "\n\n=== CURRENT DATE/TIME INFORMATION ===\n"
    datetime_info += f"Today's date: {now.strftime('%A, %B %d, %Y')}\n"
    datetime_info += f"Current time: {now.strftime('%H:%M:%S')}\n"
    datetime_info += f"Current year: {now.strftime('%Y')}\n"
    datetime_info += f"Current month: {now.strftime('%B')}\n"
    datetime_info += f"Current day: {now.strftime('%A')}\n"
    datetime_info += (
        "Use this information for any time-sensitive tasks, research, or when current date/time context is needed.\n"
    )
    content += datetime_info

    if app_config.XML_TOOL_CALLING:
        openapi_schemas = thread_manager.tool_registry.get_openapi_schemas()
        usage_examples = thread_manager.tool_registry.get_usage_examples()

        if openapi_schemas:
            # Convert schemas to JSON string
            schemas_json = json.dumps(openapi_schemas, indent=2)

            # Build usage examples section if any exist
            usage_examples_section = ""
            if usage_examples:
                usage_examples_section = "\n\nUsage Examples:\n"
                for func_name, example in usage_examples.items():
                    usage_examples_section += f"\n{func_name}:\n{example}\n"

            examples_content = f"""
        In this environment you have access to a set of tools you can use to answer the user's question.

        You can invoke functions by writing a <function_calls> block like the following as part of your reply to the user:

        <function_calls>
        <invoke name="function_name">
        <parameter name="param_name">param_value</parameter>
        ...
        </invoke>
        </function_calls>

        String and scalar parameters should be specified as-is, while lists and objects should use JSON format.

        Here are the functions available in JSON Schema format:

        ```json
        {schemas_json}
        ```

        When using the tools:
        - Use the exact function names from the JSON schema above
        - Include all required parameters as specified in the schema
        - Format complex data (objects, arrays) as JSON strings within the parameter tags
        - Boolean values should be "true" or "false" (lowercase)
        {usage_examples_section}"""

            content += examples_content

    return content


async def stream_graph_updates(user_input: str, thread_manager: ThreadManager, config):
    system_prompt = await get_system_prompt(thread_manager)

    # async with AsyncPostgresSaver.from_conn_string(app_config.SQLALCHEMY_DATABASE_URI) as checkpointer:
    with InMemorySaver() as checkpointer:
        graph = get_graph(thread_manager)

        async for event in graph.astream(
            {
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}],
                "native_max_auto_continues": 25,
                "max_iterations": 100,
                "enable_thinking": True,
                "reasoning_effort": "low",
                "llm_max_tokens": 8192,
            },
            config,
            stream_mode="values",
        ):
            if "messages" in event:
                logger.info(event["messages"][-1].pretty_print())
