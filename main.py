from typing import Annotated, Optional, TypedDict

from langchain_core.tools import Tool, tool
from langchain_litellm import ChatLiteLLM
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt
from loguru import logger

from configs import app_config
from tools.base_tool import BaseTool
from tools.message_tool import MessageTool
from tools.task_list_tool import TaskListTool
from tools.tool_registry import ToolRegistry

tool_registry = ToolRegistry()


def add_tool(tool_class: type[BaseTool], function_names: Optional[list[str]] = None, **kwargs):
    """Add a tool to the ThreadManager."""
    tool_registry.register_tool(tool_class, function_names, **kwargs)


add_tool(MessageTool)
add_tool(TaskListTool, thread_id="1")


def get_core_tools() -> list[Tool]:
    functions = tool_registry.get_available_functions()
    tools = []

    for name, func in functions.items():
        # Tool(name=,func=,description=)
        desc = tool_registry.get_tool(name)["schema"].schema["function"]["description"]
        params = tool_registry.get_tool(name)["schema"].schema["function"]["parameters"]
        tool = Tool(name=name, func=func, description=desc, args_schema=params)
        tools.append(tool)

    return tools


llm = ChatLiteLLM(
    model=app_config.model_name,
    api_base=app_config.base_url,
    api_key=app_config.api_key,
    custom_llm_provider="openai",
)


@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]


# llm = ChatOpenAI(model="Qwen/Qwen3-14B", base_url=app_config.base_url, api_key=app_config.api_key)
tool = TavilySearch(tavily_api_key=app_config.tavily_api_key, max_results=app_config.max_results)
tools = [tool, human_assistance] + get_core_tools()
llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def get_graph() -> CompiledStateGraph:
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )

    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    memory = InMemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    return graph


def get_system_prompt() -> str:
    with open("prompt.md", encoding="utf-8") as f:
        content = f.read()

    return content


async def stream_graph_updates(user_input: str, config):
    system_prompt = get_system_prompt()
    graph = get_graph()

    async for event in graph.astream(
        {"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    ):
        if "messages" in event:
            logger.info(event["messages"][-1].pretty_print())


if __name__ == "__main__":
    import asyncio

    async def run():
        user_input = "今天厦门和福州的温度谁更高"
        config = {"configurable": {"thread_id": "1"}}

        await stream_graph_updates(user_input, config)

    asyncio.run(run())
