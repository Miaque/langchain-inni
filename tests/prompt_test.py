import datetime

import pytest
from loguru import logger

from utils.current_date import get_current_date_info


@pytest.mark.asyncio
async def test_time():
    now = datetime.datetime.now()
    datetime_info = f"\n\n=== CURRENT DATE/TIME INFORMATION ===\n"
    datetime_info += f"Today's date: {now.strftime('%A, %B %d, %Y')}\n"
    datetime_info += f"Current UTC time: {now.strftime('%H:%M:%S UTC')}\n"
    datetime_info += f"Current year: {now.strftime('%Y')}\n"
    datetime_info += f"Current month: {now.strftime('%B')}\n"
    datetime_info += f"Current day: {now.strftime('%A')}\n"
    datetime_info += "Use this information for any time-sensitive tasks, research, or when current date/time context is needed.\n"
    logger.info(datetime_info)

    logger.info(get_current_date_info())