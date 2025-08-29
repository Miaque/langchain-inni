from pydantic import Field
from pydantic_settings import BaseSettings


class SandboxConfig(BaseSettings):
    daytona_snapshot_name: str = Field(default="kortix/suna:0.1.3.11")
    daytona_api_key: str = Field(default="")
    daytona_api_url: str = Field(default="")
    daytona_target: str = Field(default="us")
