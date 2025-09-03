import sys
import uuid

import pytest
from loguru import logger

from models.projects import Projects
from models.threads import Threads
from tools.sandbox.sandbox import create_sandbox, delete_sandbox

logger.remove()
logger.add(sys.stderr, level="INFO")


@pytest.mark.asyncio
async def test_create_thread():
    project_name = "New Project"
    account_id = str(uuid.uuid4())

    # 1. Create Project
    project = Projects.insert(
        {
            "project_id": str(uuid.uuid4()),
            "account_id": account_id,
            "name": project_name,
        }
    )
    project_id = project.project_id
    logger.info("Created new project: {}", project_id)

    # 2. Create Sandbox
    sandbox_id = None
    try:
        sandbox_pass = str(uuid.uuid4())
        sandbox = await create_sandbox(sandbox_pass, project_id)
        sandbox_id = sandbox.id
        logger.info(f"Created new sandbox {sandbox_id} for project {project_id}")

        # Get preview links
        vnc_link = await sandbox.get_preview_link(6080)
        website_link = await sandbox.get_preview_link(8080)
        vnc_url = vnc_link.url if hasattr(vnc_link, "url") else str(vnc_link).split("url='")[1].split("'")[0]
        website_url = (
            website_link.url if hasattr(website_link, "url") else str(website_link).split("url='")[1].split("'")[0]
        )
        token = None
        if hasattr(vnc_link, "token"):
            token = vnc_link.token
        elif "token='" in str(vnc_link):
            token = str(vnc_link).split("token='")[1].split("'")[0]
    except Exception as e:
        logger.error(f"Error creating sandbox: {str(e)}")
        Projects.delete_by_id(project_id)
        if sandbox_id:
            try:
                await delete_sandbox(sandbox_id)
            except Exception as e:
                logger.error(f"Error deleting sandbox: {str(e)}")
        raise Exception("Failed to create sandbox")

    update_result = Projects.update(
        project_id,
        {
            "sandbox": {
                "id": sandbox_id,
                "pass": sandbox_pass,
                "vnc_preview": vnc_url,
                "sandbox_url": website_url,
                "token": token,
            }
        },
    )

    if not update_result:
        logger.error(f"Failed to update project {project_id} with new sandbox {sandbox_id}")
        if sandbox_id:
            try:
                await delete_sandbox(sandbox_id)
            except Exception as e:
                logger.error(f"Error deleting sandbox: {str(e)}")
        raise Exception("Database update failed")

    # 3. Create Thread
    thread_data = {
        "thread_id": str(uuid.uuid4()),
        "project_id": project_id,
        "account_id": account_id,
    }

    thread = Threads.insert(thread_data)
    thread_id = thread.thread_id

    logger.info(f"Created new thread: {thread_id}")
    logger.info(f"Successfully created thread {thread_id} with project {project_id}")
