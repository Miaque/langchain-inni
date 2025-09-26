from pydantic import BaseModel, Field


class WebSearchConfig(BaseModel):
    MAX_RESULTS: int = Field(default=5)
    TAVILY_API_KEY: str = Field(default="")

    FIRECRAWL_API_KEY: str = Field(default="")
    FIRECRAWL_URL: str = Field(default="https://api.firecrawl.dev")