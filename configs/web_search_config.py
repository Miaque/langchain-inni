from pydantic import BaseModel, Field


class WebSearchConfig(BaseModel):
    max_results: int = Field(default=5)
    tavily_api_key: str = Field(default="")

    firecrawl_api_key: str = Field(default="")
    firecrawl_url: str = Field(default="https://api.firecrawl.dev")