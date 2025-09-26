from pydantic import Field
from pydantic_settings import BaseSettings


class LangfuseConfig(BaseSettings):
    LANGFUSE_PUBLIC_KEY: str = Field(description="langfuse public key", default="")

    LANGFUSE_SECRET_KEY: str = Field(description="langfuse secret key", default="")

    LANGFUSE_HOST: str = Field(description="langfuse host", default="")
