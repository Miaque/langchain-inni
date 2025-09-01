from datetime import datetime

from pydantic import BaseModel
from sqlalchemy import TIMESTAMP, UUID, Boolean, Column, Text, text
from sqlalchemy.dialects.postgresql import JSONB

from storage.db import Base


class Message(Base):
    __tablename__ = "project"

    message_id = Column(UUID, primary_key=True, server_default=text("gen_random_uuid()"))
    thread_id = Column(UUID, nullable=False)
    type = Column(Text, nullable=False)
    is_llm_message = Column(Boolean, nullable=False, default=True)
    content = Column(JSONB, nullable=False)
    metadata = Column(JSONB, nullable=False, default={})
    created_at = Column(TIMESTAMP, nullable=False, default=text("TIMEZONE('Asia/Shanghai'::text, NOW())"))
    updated_at = Column(TIMESTAMP, nullable=False, default=text("TIMEZONE('Asia/Shanghai'::text, NOW())"))


class MessageModel(BaseModel):
    message_id: str
    thread_id: str
    type: str
    is_llm_message: bool
    content: dict
    metadata: dict
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
