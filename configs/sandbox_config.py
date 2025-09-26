from pydantic import Field
from pydantic_settings import BaseSettings


class SandboxConfig(BaseSettings):
    DAYTONA_SNAPSHOT_NAME: str = Field(default="kortix/suna:0.1.3.11")
    DAYTONA_API_KEY: str = Field(default="")
    DAYTONA_API_URL: str = Field(default="")
    DAYTONA_TARGET: str = Field(default="us")
