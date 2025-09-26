from typing import Literal

from pydantic import Field, NonNegativeInt
from pydantic_settings import BaseSettings

# Type alias for XML result adding strategy
XmlAddingStrategy = Literal["user_message", "assistant_message", "inline_edit"]

# Type alias for tool execution strategy
ToolExecutionStrategy = Literal["sequential", "parallel"]


class ProcessorConfig(BaseSettings):
    XML_TOOL_CALLING: bool = Field(description="Enable XML-based tool call detection (<tool>...</tool>)", default=True)

    NATIVE_TOOL_CALLING: bool = Field(description="Enable OpenAI-style function calling format", default=False)

    EXECUTE_TOOLS: bool = Field(description="Whether to automatically execute detected tool calls", default=True)

    EXECUTE_ON_STREAM: bool = Field(
        description="For streaming, execute tools as they appear vs. at the end", default=False
    )

    TOOL_EXECUTION_STRATEGY: ToolExecutionStrategy = Field(
        description='How to execute multiple tools ("sequential" or "parallel")', default="sequential"
    )

    XML_ADDING_STRATEGY: XmlAddingStrategy = Field(
        description="How to add XML tool results to the conversation", default="assistant_message"
    )

    MAX_XML_TOOL_CALLS: NonNegativeInt = Field(
        description="Maximum number of XML tool calls to process (0 = no limit)", default=0
    )
