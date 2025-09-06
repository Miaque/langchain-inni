from typing import Literal

from pydantic import Field, NonNegativeInt
from pydantic_settings import BaseSettings

# Type alias for XML result adding strategy
XmlAddingStrategy = Literal["user_message", "assistant_message", "inline_edit"]

# Type alias for tool execution strategy
ToolExecutionStrategy = Literal["sequential", "parallel"]


class ProcessorConfig(BaseSettings):
    xml_tool_calling: bool = Field(description="Enable XML-based tool call detection (<tool>...</tool>)", default=True)

    native_tool_calling: bool = Field(description="Enable OpenAI-style function calling format", default=False)

    execute_tools: bool = Field(description="Whether to automatically execute detected tool calls", default=True)

    execute_on_stream: bool = Field(
        description="For streaming, execute tools as they appear vs. at the end", default=False
    )

    tool_execution_strategy: ToolExecutionStrategy = Field(
        description='How to execute multiple tools ("sequential" or "parallel")', default="sequential"
    )

    xml_adding_strategy: XmlAddingStrategy = Field(
        description="How to add XML tool results to the conversation", default="assistant_message"
    )

    max_xml_tool_calls: NonNegativeInt = Field(
        description="Maximum number of XML tool calls to process (0 = no limit)", default=0
    )
