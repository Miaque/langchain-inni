from datetime import datetime

from pydantic import BaseModel
from sqlalchemy import TIMESTAMP, UUID, Boolean, Column, Text, text
from sqlalchemy.dialects.postgresql import JSONB

from storage.db import Base


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
