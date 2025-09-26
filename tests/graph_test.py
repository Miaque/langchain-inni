from typing import Annotated, TypedDict

import pytest
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.constants import START
from langgraph.graph import MessagesState, StateGraph, add_messages
from langgraph.types import Command
from loguru import logger

from configs import app_config
from graph import graph_builder


def test_graph_init():
    with PostgresSaver.from_conn_string(app_config.SQLALCHEMY_DATABASE_URI) as checkpointer:
        checkpointer.setup()


@pytest.mark.asyncio
async def test_graph_run():
    user_input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"
    config = {"configurable": {"thread_id": "2"}}

    with PostgresSaver.from_conn_string(app_config.SQLALCHEMY_DATABASE_URI) as checkpointer:
        graph = graph_builder.compile(checkpointer=checkpointer)

        event = graph.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
            stream_mode="values",
        )

        if "messages" in event:
            event["messages"][-1].pretty_print()


def test_graph_state():
    config = {"configurable": {"thread_id": "2"}}

    with PostgresSaver.from_conn_string(app_config.SQLALCHEMY_DATABASE_URI) as checkpointer:
        graph = graph_builder.compile(checkpointer=checkpointer)
        snapshot = graph.get_state(config)

        logger.info("图节点快照信息: {}", snapshot)


def test_resume():
    config = {"configurable": {"thread_id": "2"}}

    human_response = (
        "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
        " It's much more reliable and extensible than simple autonomous agents."
    )

    human_command = Command(resume={"data": human_response})
    with PostgresSaver.from_conn_string(app_config.SQLALCHEMY_DATABASE_URI) as checkpointer:
        graph = graph_builder.compile(checkpointer=checkpointer)
        event = graph.invoke(human_command, config, stream_mode="values")

        if "messages" in event:
            event["messages"][-1].pretty_print()


def test_graph_history():
    config = {"configurable": {"thread_id": "2"}}
    with PostgresSaver.from_conn_string(app_config.SQLALCHEMY_DATABASE_URI) as checkpointer:
        graph = graph_builder.compile(checkpointer=checkpointer)
        history = graph.get_state_history(config)

        for event in history:
            if "messages" in event:
                event["messages"][-1].pretty_print()


def test_subgraph():
    class SubgraphMessagesState(TypedDict):
        subgraph_messages: Annotated[list, add_messages]

    def call_model(state: SubgraphMessagesState):
        response = llm.invoke(state["subgraph_messages"])
        return {"subgraph_messages": response}

    subgraph_builder = StateGraph(SubgraphMessagesState)
    subgraph_builder.add_node("call_model_from_subgraph", call_model)
    subgraph_builder.add_edge(START, "call_model_from_subgraph")

    subgraph = subgraph_builder.compile()

    def call_subgraph(state: MessagesState):
        response = subgraph.invoke({"subgraph_messages": state["messages"]})
        return {"messages": response["subgraph_messages"]}
