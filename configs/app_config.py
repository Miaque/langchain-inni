from pydantic_settings import SettingsConfigDict

from configs.data_provider_config import DataProviderConfig
from configs.database_config import DatabaseConfig
from configs.llm_config import LLMConfig
from configs.processor_config import ProcessorConfig
from configs.sandbox_config import SandboxConfig
from configs.web_search_config import WebSearchConfig


class AppConfig(LLMConfig, WebSearchConfig, DatabaseConfig, SandboxConfig, DataProviderConfig, ProcessorConfig):
    model_config = SettingsConfigDict(
        # read from dotenv format config file
        env_file=".env",
        env_file_encoding="utf-8",
        # ignore extra attributes
        extra="ignore",
    )
