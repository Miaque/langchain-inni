from pydantic import Field
from pydantic_settings import BaseSettings


class DataProviderConfig(BaseSettings):
    RAPID_API_KEY: str = Field(default="")