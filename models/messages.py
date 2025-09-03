import uuid
from datetime import datetime
from typing import Optional

from loguru import logger
from pydantic import BaseModel
from sqlalchemy import TIMESTAMP, UUID, Boolean, Column, Text, text
from sqlalchemy.dialects.postgresql import JSONB

from storage.db import Base, get_db


class Message(Base):
    __tablename__ = "message"

    message_id = Column(UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()"))
    thread_id = Column(UUID(as_uuid=False), nullable=False)
    type = Column(Text, nullable=False)
    is_llm_message = Column(Boolean, nullable=False, default=True)
    content = Column(JSONB, nullable=False)
    meta_data = Column(JSONB, nullable=False, default={})
    created_at = Column(TIMESTAMP, nullable=False, default=text("TIMEZONE('Asia/Shanghai'::text, NOW())"))
    updated_at = Column(TIMESTAMP, nullable=False, default=text("TIMEZONE('Asia/Shanghai'::text, NOW())"))


class MessageModel(BaseModel):
    message_id: Optional[str] = None
    thread_id: str
    type: str
    is_llm_message: bool
    content: dict
    meta_data: dict
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class MessageTable:
    @staticmethod
    def get_last_message(thread_id: str, type: str) -> Optional[MessageModel]:
        try:
            with get_db() as db:
                message = (
                    db.query(Message)
                    .filter(Message.thread_id == thread_id)
                    .filter(Message.type == type)
                    .order_by(Message.created_at.desc())
                    .first()
                )
                return MessageModel.model_validate(message) if message else None
        except Exception as e:
            logger.exception("Error getting last message: ", exc_info=e)
            return None

    @staticmethod
    def get_last_message_id(thread_id: str, type: str) -> Optional[str]:
        try:
            with get_db() as db:
                message = (
                    db.query(Message.message_id)
                    .filter(Message.thread_id == thread_id)
                    .filter(Message.type == type)
                    .order_by(Message.created_at.desc())
                    .first()
                )
            return message[0] if message else None
        except Exception as e:
            logger.exception("Error getting last message id: ", exc_info=e)
            return None

    @staticmethod
    def update_content(message_id: str, content: dict) -> bool:
        try:
            with get_db() as db:
                result = db.query(Message).filter(Message.message_id == message_id).update({"content": content})
                db.commit()
                return True if result == 1 else False
        except Exception as e:
            logger.exception("Error updating content: ", exc_info=e)
            return False

    @staticmethod
    def save_message(msg: dict):
        try:
            with get_db() as db:
                message = Message(**{**msg, "message_id": str(uuid.uuid4())})
                db.add(message)
                db.commit()
        except Exception as e:
            logger.exception("Error saving message: ", exc_info=e)
            raise e


Messages = MessageTable()
