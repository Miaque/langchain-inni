from datetime import datetime
from typing import Optional

from loguru import logger
from pydantic import BaseModel
from sqlalchemy import TIMESTAMP, UUID, Boolean, Column, text

from storage.db import Base, get_db


class Thread(Base):
    __tablename__ = "thread"

    thread_id = Column(UUID(as_uuid=False), primary_key=True, server_default=text("gen_random_uuid()"))
    account_id = Column(UUID(as_uuid=False), nullable=False)
    project_id = Column(UUID(as_uuid=False), nullable=False)
    is_public = Column(Boolean, nullable=False, default=False)
    created_at = Column(TIMESTAMP, nullable=False, default=text("TIMEZONE('Asia/Shanghai'::text, NOW())"))
    updated_at = Column(TIMESTAMP, nullable=False, default=text("TIMEZONE('Asia/Shanghai'::text, NOW())"))


class ThreadModel(BaseModel):
    thread_id: str
    account_id: str
    project_id: str
    is_public: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ThreadTable:
    @staticmethod
    def insert(thread: dict) -> Optional[ThreadModel]:
        try:
            with get_db() as db:
                thread = Thread(**thread)
                db.add(thread)
                db.commit()
                return ThreadModel.model_validate(thread)
        except Exception as e:
            logger.exception("Error inserting thread: ", exc_info=e)
            return None


Threads = ThreadTable()
