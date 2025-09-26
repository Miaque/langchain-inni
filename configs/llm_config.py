from pydantic import Field, computed_field
from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    BASE_URL: str = Field(description="llm base url", default="https://api.siliconflow.cn/v1")

    API_KEY: str = Field(description="llm api key", default="")

    MODEL_NAME: str = Field(description="llm model name", default="Qwen/Qwen3-Next-80B-A3B-Instruct")

    REASON_MODEL_NAME: str = Field(description="llm reason model name", default="Qwen/Qwen3-Next-80B-A3B-Thinking")

    @computed_field
    @property
    def DEFAULT_MODEL(self) -> str:
        return self.MODEL_NAME

    @computed_field
    @property
    def INSTRUCT_MODEL_NAME(self) -> str:
        return self.MODEL_NAME
