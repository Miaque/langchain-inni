import sys
import uuid

import pytest
from loguru import logger

from models.projects import Projects

logger.remove()
logger.add(sys.stderr, level="INFO")


@pytest.mark.asyncio
async def test_create_project():
    project = {
        "project_id": str(uuid.uuid4()),
        "account_id": str(uuid.uuid4()),
        "name": "test_project",
    }

    result = Projects.insert(project)
    logger.info(result)


@pytest.mark.asyncio
async def test_update_project():
    project = {
        "sandbox": {
            "id": str(uuid.uuid4()),
            "pass": str(uuid.uuid4()),
            "vnc_preview": "vnc_url",
            "sandbox_url": "website_url",
            "token": "token",
        }
    }

    result = Projects.update("f32b9480-e27f-4f99-a070-0510306461e5", project)
    logger.info(result)
