import pytest
from loguru import logger

from llm import get_instruct_llm, get_reason_llm


@pytest.mark.asyncio
async def test_instruct_llm():
    llm = get_instruct_llm()
    response = await llm.ainvoke("你好，你是谁？")
    logger.info(response)


@pytest.mark.asyncio
async def test_reason_llm():
    llm = get_reason_llm()
    response = await llm.ainvoke("你好，你是谁？")
    logger.info(response)
