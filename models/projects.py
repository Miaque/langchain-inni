from datetime import datetime
from typing import Optional

from loguru import logger
from pydantic import BaseModel
from sqlalchemy import TIMESTAMP, UUID, Boolean, Column, Text, text
from sqlalchemy.dialects.postgresql import JSONB

from storage.db import Base, get_db


class Project(Base):
    __tablename__ = "project"

    project_id = Column(UUID, primary_key=True, server_default=text("gen_random_uuid()"))
    name = Column(Text, nullable=False)
    account_id = Column(UUID, nullable=False)
    sandbox = Column(JSONB, nullable=False, default={})
    is_public = Column(Boolean, nullable=False, default=False)
    created_at = Column(TIMESTAMP, nullable=False, default=text("TIMEZONE('Asia/Shanghai'::text, NOW())"))
    updated_at = Column(TIMESTAMP, nullable=False, default=text("TIMEZONE('Asia/Shanghai'::text, NOW())"))


class ProjectModel(BaseModel):
    project_id: str
    name: str
    account_id: str
    sandbox: dict
    is_public: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ProjectTable:
    @staticmethod
    def get_by_id(project_id: str) -> Optional[ProjectModel]:
        try:
            with get_db() as db:
                project = db.query(Project).filter(Project.project_id == project_id).first()
                return ProjectModel.model_validate(project)
        except Exception as e:
            logger.exception("Error getting project by id: ", exc_info=e)
            return None

    @staticmethod
    def update_sandbox(project_id: str, sandbox: dict) -> bool:
        try:
            with get_db() as db:
                result = db.query(Project).filter(Project.project_id == project_id).update({"sandbox": sandbox})
                db.commit()

                return True if result == 1 else False
        except Exception as e:
            logger.exception("Error updating sandbox: ", exc_info=e)
            return False


Projects = ProjectTable()
