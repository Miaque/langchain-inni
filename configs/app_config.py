from pydantic_settings import SettingsConfigDict

from configs.database_config import DatabaseConfig
from configs.llm_config import LLMConfig
from configs.web_search_config import WebSearchConfig


class AppConfig(LLMConfig, WebSearchConfig, DatabaseConfig):
    model_config = SettingsConfigDict(
        # read from dotenv format config file
        env_file=".env",
        env_file_encoding="utf-8",
        # ignore extra attributes
        extra="ignore",
    )