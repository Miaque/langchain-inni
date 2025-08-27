from pydantic import Field
from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    base_url: str = Field(description="llm base url", default="https://api.siliconflow.cn/v1")
    api_key: str = Field(description="llm api key", default="")
    model_name: str = Field(description="llm model name", default="moonshotai/Kimi-K2-Instruct")
