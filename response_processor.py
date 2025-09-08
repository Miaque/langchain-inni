import asyncio
import json
import uuid
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any, Optional, Union

from loguru import logger

from configs import app_config
from configs.processor_config import ToolExecutionStrategy, XmlAddingStrategy
from models.messages import MessageModel
from tools.base_tool import ToolResult
from tools.tool_registry import ToolRegistry
from utils.json_helpers import format_for_yield, safe_json_parse, to_json_string
from utils.xml_tool_parser import XMLToolParser


@dataclass
class ToolExecutionContext:
    """工具执行的上下文，包括调用详情、结果和显示信息。"""

    tool_call: dict[str, Any]
    tool_index: int
    result: Optional[ToolResult] = None
    function_name: Optional[str] = None
    xml_tag_name: Optional[str] = None
    error: Optional[Exception] = None
    assistant_message_id: Optional[str] = None
    parsing_details: Optional[dict[str, Any]] = None


class ResponseProcessor:
    """处理LLM响应，提取并执行工具调用。"""

    def __init__(
        self, tool_registry: ToolRegistry, add_message_callback: Callable, agent_config: Optional[dict] = None
    ):
        """初始化ResponseProcessor。

        参数:
            tool_registry: 可用工具的注册表
            add_message_callback: 向线程添加消息的回调函数。
                必须返回完整的已保存消息对象（dict）或None。
            agent_config: 包含版本信息的可选代理配置
        """
        self.tool_registry = tool_registry
        self.add_message = add_message_callback
        # 初始化XML解析器
        self.xml_parser = XMLToolParser()
        self.is_agent_builder = False  # 已弃用 - 保持兼容性
        self.target_agent_id = None  # 已弃用 - 保持兼容性
        self.agent_config = agent_config

    @staticmethod
    async def _yield_message(message_obj: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """帮助函数，以正确格式生成消息。

        确保内容和元数据是JSON字符串，以便客户端兼容。
        """
        if message_obj:
            return format_for_yield(message_obj)
        return None

    @staticmethod
    def _serialize_model_response(model_response) -> dict[str, Any]:
        """将LiteLLM ModelResponse对象转换为可JSON序列化的字典。

        参数:
            model_response: LiteLLM ModelResponse对象

        返回:
            ModelResponse的字典表示
        """
        try:
            # 尝试使用model_dump方法（如果可用）（Pydantic v2）
            if hasattr(model_response, "model_dump"):
                return model_response.model_dump()

            # 尝试使用dict方法（如果可用）（Pydantic v1）
            elif hasattr(model_response, "dict"):
                return model_response.dict()

            # 回退：手动提取常见属性
            else:
                result = {}

                # 常见的LiteLLM ModelResponse属性
                for attr in ["id", "object", "created", "model", "choices", "usage", "system_fingerprint"]:
                    if hasattr(model_response, attr):
                        value = getattr(model_response, attr)
                        # 递归处理嵌套对象
                        if hasattr(value, "model_dump"):
                            result[attr] = value.model_dump()
                        elif hasattr(value, "dict"):
                            result[attr] = value.dict()
                        elif isinstance(value, list):
                            result[attr] = [
                                item.model_dump()
                                if hasattr(item, "model_dump")
                                else item.dict()
                                if hasattr(item, "dict")
                                else item
                                for item in value
                            ]
                        else:
                            result[attr] = value

                return result

        except Exception as e:
            logger.warning(f"序列化ModelResponse失败: {str(e)}，回退到字符串表示")
            # 最终回退：转换为字符串
            return {"raw_response": str(model_response), "serialization_error": str(e)}

    async def _add_message_with_agent_info(
        self,
        thread_id: str,
        type: str,
        content: Union[dict[str, Any], list[Any], str],
        is_llm_message: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """帮助函数，如果可用则添加带有代理版本信息的消息。"""
        agent_id = None
        agent_version_id = None

        if self.agent_config:
            agent_id = self.agent_config.get("agent_id")
            agent_version_id = self.agent_config.get("current_version_id")

        return await self.add_message(
            thread_id=thread_id,
            type=type,
            content=content,
            is_llm_message=is_llm_message,
            metadata=metadata,
            agent_id=agent_id,
            agent_version_id=agent_version_id,
        )

    async def process_non_streaming_response(
        self,
        llm_response: Any,
        thread_id: str,
        prompt_messages: Optional[list[dict[str, Any]]] = None,
        llm_model: Optional[str] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Process a non-streaming LLM response, handling tool calls and execution.

        Args:
            llm_response: Response from the LLM
            thread_id: ID of the conversation thread
            prompt_messages: List of messages sent to the LLM (the prompt)
            llm_model: The name of the LLM model used

        Yields:
            Complete message objects matching the DB schema.
        """
        content = ""
        thread_run_id = str(uuid.uuid4())
        all_tool_data = []  # Stores {'tool_call': ..., 'parsing_details': ...}
        tool_index = 0
        assistant_message_object = None
        tool_result_message_objects = {}
        finish_reason = None
        native_tool_calls_for_message = []

        try:
            # Save and Yield thread_run_start status message
            start_content = {"status_type": "thread_run_start", "thread_run_id": thread_run_id}
            start_msg_obj = await self.add_message(
                thread_id=thread_id,
                type="status",
                content=start_content,
                is_llm_message=False,
                metadata={"thread_run_id": thread_run_id},
            )
            if start_msg_obj:
                yield format_for_yield(start_msg_obj)

            # Extract finish_reason, content, tool calls
            if hasattr(llm_response, "response_metadata") and llm_response.response_metadata:
                if "finish_reason" in llm_response.response_metadata:
                    finish_reason = llm_response.response_metadata["finish_reason"]
                    logger.debug(f"非流式finish_reason: {finish_reason}")

                if hasattr(llm_response, "content") and llm_response.content:
                    content = llm_response.content
                    if app_config.xml_tool_calling:
                        parsed_xml_data = self._parse_xml_tool_calls(content)
                        if 0 < app_config.max_xml_tool_calls < len(parsed_xml_data):
                            # Truncate content and tool data if limit exceeded
                            # ... (Truncation logic similar to streaming) ...
                            if parsed_xml_data:
                                xml_chunks = self._extract_xml_chunks(content)[: app_config.max_xml_tool_calls]
                                if xml_chunks:
                                    last_chunk = xml_chunks[-1]
                                    last_chunk_pos = content.find(last_chunk)
                                    if last_chunk_pos >= 0:
                                        content = content[: last_chunk_pos + len(last_chunk)]
                            parsed_xml_data = parsed_xml_data[: app_config.max_xml_tool_calls]
                            finish_reason = "xml_tool_limit_reached"
                        all_tool_data.extend(parsed_xml_data)

                if app_config.native_tool_calling and hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
                    for tool_call in llm_response.tool_calls:
                        if hasattr(tool_call, "function"):
                            exec_tool_call = {
                                "function_name": tool_call.function.name,
                                "arguments": safe_json_parse(tool_call.function.arguments)
                                if isinstance(tool_call.function.arguments, str)
                                else tool_call.function.arguments,
                                "id": tool_call.id if hasattr(tool_call, "id") else str(uuid.uuid4()),
                            }
                            all_tool_data.append({"tool_call": exec_tool_call, "parsing_details": None})
                            native_tool_calls_for_message.append(
                                {
                                    "id": exec_tool_call["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments
                                        if isinstance(tool_call.function.arguments, str)
                                        else to_json_string(tool_call.function.arguments),
                                    },
                                }
                            )

            # --- SAVE and YIELD Final Assistant Message ---
            message_data = {
                "role": "assistant",
                "content": content,
                "tool_calls": native_tool_calls_for_message or None,
            }
            assistant_message_object = await self._add_message_with_agent_info(
                thread_id=thread_id,
                type="assistant",
                content=message_data,
                is_llm_message=True,
                metadata={"thread_run_id": thread_run_id},
            )
            if assistant_message_object:
                yield assistant_message_object
            else:
                logger.error(f"Failed to save non-streaming assistant message for thread {thread_id}")
                err_content = {"role": "system", "status_type": "error", "message": "Failed to save assistant message"}
                err_msg_obj = await self.add_message(
                    thread_id=thread_id,
                    type="status",
                    content=err_content,
                    is_llm_message=False,
                    metadata={"thread_run_id": thread_run_id},
                )
                if err_msg_obj:
                    yield format_for_yield(err_msg_obj)

            # --- Execute Tools and Yield Results ---
            tool_calls_to_execute = [item["tool_call"] for item in all_tool_data]
            if app_config.execute_tools and tool_calls_to_execute:
                logger.debug(f"使用策略执行 {len(tool_calls_to_execute)} 个工具: {app_config.tool_execution_strategy}")
                tool_results = await self._execute_tools(tool_calls_to_execute, app_config.tool_execution_strategy)

                for i, (returned_tool_call, result) in enumerate(tool_results):
                    original_data = all_tool_data[i]
                    tool_call_from_data = original_data["tool_call"]
                    parsing_details = original_data["parsing_details"]
                    current_assistant_id = assistant_message_object.message_id if assistant_message_object else None

                    context = self._create_tool_context(
                        tool_call_from_data, tool_index, current_assistant_id, parsing_details
                    )
                    context.result = result

                    # Save and Yield start status
                    started_msg_obj = await self._yield_and_save_tool_started(context, thread_id, thread_run_id)
                    if started_msg_obj:
                        yield format_for_yield(started_msg_obj)

                    # Save tool result
                    saved_tool_result_object = await self._add_tool_result(
                        thread_id,
                        tool_call_from_data,
                        result,
                        app_config.xml_adding_strategy,
                        current_assistant_id,
                        parsing_details,
                    )

                    # Save and Yield completed/failed status
                    completed_msg_obj = await self._yield_and_save_tool_completed(
                        context,
                        saved_tool_result_object.message_id if saved_tool_result_object else None,
                        thread_id,
                        thread_run_id,
                    )
                    if completed_msg_obj:
                        yield format_for_yield(completed_msg_obj)

                    # Yield the saved tool result object
                    if saved_tool_result_object:
                        tool_result_message_objects[tool_index] = saved_tool_result_object
                        yield format_for_yield(saved_tool_result_object)
                    else:
                        logger.error(f"Failed to save tool result for index {tool_index}")

                    tool_index += 1

            # --- Save and Yield Final Status ---
            if finish_reason:
                finish_content = {"status_type": "finish", "finish_reason": finish_reason}
                finish_msg_obj = await self.add_message(
                    thread_id=thread_id,
                    type="status",
                    content=finish_content,
                    is_llm_message=False,
                    metadata={"thread_run_id": thread_run_id},
                )
                if finish_msg_obj:
                    yield format_for_yield(finish_msg_obj)

            # --- Save and Yield assistant_response_end ---
            if assistant_message_object:  # Only save if assistant message was saved
                try:
                    # Convert LiteLLM ModelResponse to a JSON-serializable dictionary
                    response_dict = self._serialize_model_response(llm_response)

                    # Save the serialized response object in content
                    await self.add_message(
                        thread_id=thread_id,
                        type="assistant_response_end",
                        content=response_dict,
                        is_llm_message=False,
                        metadata={"thread_run_id": thread_run_id},
                    )
                    logger.debug("Assistant response end saved for non-stream")
                except Exception as e:
                    logger.error(f"Error saving assistant response end for non-stream: {str(e)}")

        except Exception as e:
            logger.error(f"Error processing non-streaming response: {str(e)}", exc_info=True)

            # Save and yield error status
            err_content = {"role": "system", "status_type": "error", "message": str(e)}
            err_msg_obj = await self.add_message(
                thread_id=thread_id,
                type="status",
                content=err_content,
                is_llm_message=False,
                metadata={"thread_run_id": thread_run_id if "thread_run_id" in locals() else None},
            )
            if err_msg_obj:
                yield format_for_yield(err_msg_obj)

            # Re-raise the same exception (not a new one) to ensure proper error propagation
            logger.critical(f"Re-raising error to stop further processing: {str(e)}")
            raise  # Use bare 'raise' to preserve the original exception with its traceback

        finally:
            # Save and Yield the final thread_run_end status
            end_content = {"status_type": "thread_run_end"}
            end_msg_obj = await self.add_message(
                thread_id=thread_id,
                type="status",
                content=end_content,
                is_llm_message=False,
                metadata={"thread_run_id": thread_run_id if "thread_run_id" in locals() else None},
            )
            if end_msg_obj:
                yield format_for_yield(end_msg_obj)

    def _extract_xml_chunks(self, content: str) -> list[str]:
        """Extract complete XML chunks using start and end pattern matching."""
        chunks = []
        pos = 0

        try:
            # First, look for new format <function_calls> blocks
            start_pattern = "<function_calls>"
            end_pattern = "</function_calls>"

            while pos < len(content):
                # Find the next function_calls block
                start_pos = content.find(start_pattern, pos)
                if start_pos == -1:
                    break

                # Find the matching end tag
                end_pos = content.find(end_pattern, start_pos)
                if end_pos == -1:
                    break

                # Extract the complete block including tags
                chunk_end = end_pos + len(end_pattern)
                chunk = content[start_pos:chunk_end]
                chunks.append(chunk)

                # Move position past this chunk
                pos = chunk_end

            # If no new format found, fall back to old format for backwards compatibility
            if not chunks:
                pos = 0
                while pos < len(content):
                    # Find the next tool tag
                    next_tag_start = -1
                    current_tag = None

                    # Find the earliest occurrence of any registered tool function name
                    # Check for available function names
                    available_functions = self.tool_registry.get_available_functions()
                    for func_name in available_functions:
                        # Convert function name to potential tag name (underscore to dash)
                        tag_name = func_name.replace("_", "-")
                        start_pattern = f"<{tag_name}"
                        tag_pos = content.find(start_pattern, pos)

                        if tag_pos != -1 and (next_tag_start == -1 or tag_pos < next_tag_start):
                            next_tag_start = tag_pos
                            current_tag = tag_name

                    if next_tag_start == -1 or not current_tag:
                        break

                    # Find the matching end tag
                    end_pattern = f"</{current_tag}>"
                    tag_stack = []
                    chunk_start = next_tag_start
                    current_pos = next_tag_start

                    while current_pos < len(content):
                        # Look for next start or end tag of the same type
                        next_start = content.find(f"<{current_tag}", current_pos + 1)
                        next_end = content.find(end_pattern, current_pos)

                        if next_end == -1:  # No closing tag found
                            break

                        if next_start != -1 and next_start < next_end:
                            # Found nested start tag
                            tag_stack.append(next_start)
                            current_pos = next_start + 1
                        else:
                            # Found end tag
                            if not tag_stack:  # This is our matching end tag
                                chunk_end = next_end + len(end_pattern)
                                chunk = content[chunk_start:chunk_end]
                                chunks.append(chunk)
                                pos = chunk_end
                                break
                            else:
                                # Pop nested tag
                                tag_stack.pop()
                                current_pos = next_end + 1

                    if current_pos >= len(content):  # Reached end without finding closing tag
                        break

                    pos = max(pos + 1, current_pos)

        except Exception as e:
            logger.error(f"Error extracting XML chunks: {e}")
            logger.error(f"Content was: {content}")

        return chunks

    def _parse_xml_tool_call(self, xml_chunk: str) -> Optional[tuple[dict[str, Any], dict[str, Any]]]:
        """Parse XML chunk into tool call format and return parsing details.

        Returns:
            Tuple of (tool_call, parsing_details) or None if parsing fails.
            - tool_call: Dict with 'function_name', 'xml_tag_name', 'arguments'
            - parsing_details: Dict with 'attributes', 'elements', 'text_content', 'root_content'
        """
        try:
            # Check if this is the new format (contains <function_calls>)
            if "<function_calls>" in xml_chunk and "<invoke" in xml_chunk:
                # Use the new XML parser
                parsed_calls = self.xml_parser.parse_content(xml_chunk)

                if not parsed_calls:
                    logger.error(f"No tool calls found in XML chunk: {xml_chunk}")
                    return None

                # Take the first tool call (should only be one per chunk)
                xml_tool_call = parsed_calls[0]

                # Convert to the expected format
                tool_call = {
                    "function_name": xml_tool_call.function_name,
                    "xml_tag_name": xml_tool_call.function_name.replace("_", "-"),  # For backwards compatibility
                    "arguments": xml_tool_call.parameters,
                }

                # Include the parsing details
                parsing_details = xml_tool_call.parsing_details
                parsing_details["raw_xml"] = xml_tool_call.raw_xml

                logger.debug(f"Parsed new format tool call: {tool_call}")
                return tool_call, parsing_details

            # If not the expected <function_calls><invoke> format, return None
            logger.error(f"XML chunk does not contain expected <function_calls><invoke> format: {xml_chunk}")
            return None

        except Exception as e:
            logger.error(f"Error parsing XML chunk: {e}")
            logger.error(f"XML chunk was: {xml_chunk}")
            return None

    def _parse_xml_tool_calls(self, content: str) -> list[dict[str, Any]]:
        """Parse XML tool calls from content string.

        Returns:
            List of dictionaries, each containing {'tool_call': ..., 'parsing_details': ...}
        """
        parsed_data = []

        try:
            xml_chunks = self._extract_xml_chunks(content)

            for xml_chunk in xml_chunks:
                result = self._parse_xml_tool_call(xml_chunk)
                if result:
                    tool_call, parsing_details = result
                    parsed_data.append({"tool_call": tool_call, "parsing_details": parsing_details})

        except Exception as e:
            logger.error(f"Error parsing XML tool calls: {e}", exc_info=True)

        return parsed_data

    # Tool execution methods
    async def _execute_tool(self, tool_call: dict[str, Any]) -> ToolResult:
        """Execute a single tool call and return the result."""
        try:
            function_name = tool_call["function_name"]
            arguments = tool_call["arguments"]

            logger.debug(f"执行工具: {function_name}，参数: {arguments}")

            if isinstance(arguments, str):
                try:
                    arguments = safe_json_parse(arguments)
                except json.JSONDecodeError:
                    arguments = {"text": arguments}

            # Get available functions from tool registry
            available_functions = self.tool_registry.get_available_functions()

            # Look up the function by name
            tool_fn = available_functions.get(function_name)
            if not tool_fn:
                logger.error(f"工具函数 '{function_name}' 在注册表中未找到")
                return ToolResult(success=False, output=f"Tool function '{function_name}' not found")

            logger.debug(f"找到工具函数 '{function_name}'，正在执行...")
            result = await tool_fn(**arguments)
            logger.debug(f"工具执行完成: {function_name} -> {result}")
            return result
        except Exception as e:
            logger.error(f"执行工具 {tool_call['function_name']} 时出错: {str(e)}", exc_info=True)
            return ToolResult(success=False, output=f"Error executing tool: {str(e)}")

    async def _execute_tools(
        self, tool_calls: list[dict[str, Any]], execution_strategy: ToolExecutionStrategy = "sequential"
    ) -> list[tuple[dict[str, Any], ToolResult]]:
        """Execute tool calls with the specified strategy.

        This is the main entry point for tool execution. It dispatches to the appropriate
        execution method based on the provided strategy.

        Args:
            tool_calls: List of tool calls to execute
            execution_strategy: Strategy for executing tools:
                - "sequential": Execute tools one after another, waiting for each to complete
                - "parallel": Execute all tools simultaneously for better performance

        Returns:
            List of tuples containing the original tool call and its result
        """
        logger.debug(f"使用策略执行 {len(tool_calls)} 个工具: {execution_strategy}")

        if execution_strategy == "sequential":
            return await self._execute_tools_sequentially(tool_calls)
        elif execution_strategy == "parallel":
            return await self._execute_tools_in_parallel(tool_calls)
        else:
            logger.warning(f"Unknown execution strategy: {execution_strategy}, falling back to sequential")
            return await self._execute_tools_sequentially(tool_calls)

    async def _execute_tools_sequentially(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], ToolResult]]:
        """Execute tool calls sequentially and return results.

        This method executes tool calls one after another, waiting for each tool to complete
        before starting the next one. This is useful when tools have dependencies on each other.

        Args:
            tool_calls: List of tool calls to execute

        Returns:
            List of tuples containing the original tool call and its result
        """
        if not tool_calls:
            return []

        try:
            tool_names = [t.get("function_name", "unknown") for t in tool_calls]
            logger.debug(f"顺序执行 {len(tool_calls)} 个工具: {tool_names}")

            results = []
            for index, tool_call in enumerate(tool_calls):
                tool_name = tool_call.get("function_name", "unknown")
                logger.debug(f"执行工具 {index + 1}/{len(tool_calls)}: {tool_name}")

                try:
                    result = await self._execute_tool(tool_call)
                    results.append((tool_call, result))
                    logger.debug(f"完成工具 {tool_name}，成功={result.success}")

                    # Check if this is a terminating tool (ask or complete)
                    if tool_name in ["ask", "complete"]:
                        logger.debug(f"终止工具 '{tool_name}' 已执行。停止进一步工具执行。")
                        break  # Stop executing remaining tools

                except Exception as e:
                    logger.error(f"执行工具 {tool_name} 时出错: {str(e)}")
                    error_result = ToolResult(success=False, output=f"Error executing tool: {str(e)}")
                    results.append((tool_call, error_result))

            logger.debug(f"顺序执行完成，共 {len(results)} 个工具（总共 {len(tool_calls)} 个）")
            return results

        except Exception as e:
            logger.error(f"顺序工具执行时出错: {str(e)}", exc_info=True)
            # Return partial results plus error results for remaining tools
            completed_results = results if "results" in locals() else []
            completed_tool_names = [r[0].get("function_name", "unknown") for r in completed_results]
            remaining_tools = [t for t in tool_calls if t.get("function_name", "unknown") not in completed_tool_names]

            # Add error results for remaining tools
            error_results = [
                (tool, ToolResult(success=False, output=f"Execution error: {str(e)}")) for tool in remaining_tools
            ]

            return completed_results + error_results

    async def _execute_tools_in_parallel(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], ToolResult]]:
        """Execute tool calls in parallel and return results.

        This method executes all tool calls simultaneously using asyncio.gather, which
        can significantly improve performance when executing multiple independent tools.

        Args:
            tool_calls: List of tool calls to execute

        Returns:
            List of tuples containing the original tool call and its result
        """
        if not tool_calls:
            return []

        try:
            tool_names = [t.get("function_name", "unknown") for t in tool_calls]
            logger.debug(f"并行执行 {len(tool_calls)} 个工具: {tool_names}")

            # Create tasks for all tool calls
            tasks = [self._execute_tool(tool_call) for tool_call in tool_calls]

            # Execute all tasks concurrently with error handling
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and handle any exceptions
            processed_results = []
            for i, (tool_call, result) in enumerate(zip(tool_calls, results)):
                if isinstance(result, Exception):
                    logger.error(f"Error executing tool {tool_call.get('function_name', 'unknown')}: {str(result)}")
                    # Create error result
                    error_result = ToolResult(success=False, output=f"Error executing tool: {str(result)}")
                    processed_results.append((tool_call, error_result))
                else:
                    processed_results.append((tool_call, result))

            logger.debug(f"并行执行完成，共 {len(tool_calls)} 个工具")
            return processed_results

        except Exception as e:
            logger.error(f"并行工具执行时出错: {str(e)}", exc_info=True)
            # Return error results for all tools if the gather itself fails
            return [
                (tool_call, ToolResult(success=False, output=f"Execution error: {str(e)}")) for tool_call in tool_calls
            ]

    async def _add_tool_result(
        self,
        thread_id: str,
        tool_call: dict[str, Any],
        result: ToolResult,
        strategy: Union[XmlAddingStrategy, str] = "assistant_message",
        assistant_message_id: Optional[str] = None,
        parsing_details: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:  # Return the full message object
        """Add a tool result to the conversation thread based on the specified format.

        This method formats tool results and adds them to the conversation history,
        making them visible to the LLM in subsequent interactions. Results can be
        added either as native tool messages (OpenAI format) or as XML-wrapped content
        with a specified role (user or assistant).

        Args:
            thread_id: ID of the conversation thread
            tool_call: The original tool call that produced this result
            result: The result from the tool execution
            strategy: How to add XML tool results to the conversation
                     ("user_message", "assistant_message", or "inline_edit")
            assistant_message_id: ID of the assistant message that generated this tool call
            parsing_details: Detailed parsing info for XML calls (attributes, elements, etc.)
        """
        try:
            message_obj = None  # Initialize message_obj

            # Create metadata with assistant_message_id if provided
            metadata = {}
            if assistant_message_id:
                metadata["assistant_message_id"] = assistant_message_id
                logger.debug(f"Linking tool result to assistant message: {assistant_message_id}")

            # --- Add parsing details to metadata if available ---
            if parsing_details:
                metadata["parsing_details"] = parsing_details
                logger.debug("Adding parsing_details to tool result metadata")
            # ---

            # Check if this is a native function call (has id field)
            if "id" in tool_call:
                # Format as a proper tool message according to OpenAI spec
                function_name = tool_call.get("function_name", "")

                # Format the tool result content - tool role needs string content
                if isinstance(result, str):
                    content = result
                elif hasattr(result, "output"):
                    # If it's a ToolResult object
                    if isinstance(result.output, (dict, list)):
                        # If output is already a dict or list, convert to JSON string
                        content = json.dumps(result.output)
                    else:
                        # Otherwise just use the string representation
                        content = str(result.output)
                else:
                    # Fallback to string representation of the whole result
                    content = str(result)

                logger.debug(f"Formatted tool result content: {content[:100]}...")

                # Create the tool response message with proper format
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": function_name,
                    "content": content,
                }

                logger.debug(f"Adding native tool result for tool_call_id={tool_call['id']} with role=tool")

                # Add as a tool message to the conversation history
                # This makes the result visible to the LLM in the next turn
                message_obj = await self.add_message(
                    thread_id=thread_id,
                    type="tool",  # Special type for tool responses
                    content=tool_message,
                    is_llm_message=True,
                    metadata=metadata,
                )
                return message_obj  # Return the full message object

            # For XML and other non-native tools, use the new structured format
            # Determine message role based on strategy
            result_role = "user" if strategy == "user_message" else "assistant"

            # Create two versions of the structured result
            # 1. Rich version for the frontend
            structured_result_for_frontend = self._create_structured_tool_result(
                tool_call, result, parsing_details, for_llm=False
            )
            # 2. Concise version for the LLM
            structured_result_for_llm = self._create_structured_tool_result(
                tool_call, result, parsing_details, for_llm=True
            )

            # Add the message with the appropriate role to the conversation history
            # This allows the LLM to see the tool result in subsequent interactions
            result_message_for_llm = {"role": result_role, "content": json.dumps(structured_result_for_llm)}

            # Add rich content to metadata for frontend use
            if metadata is None:
                metadata = {}
            metadata["frontend_content"] = structured_result_for_frontend

            message_obj = await self._add_message_with_agent_info(
                thread_id=thread_id,
                type="tool",
                content=result_message_for_llm,  # Save the LLM-friendly version
                is_llm_message=True,
                metadata=metadata,
            )

            # If the message was saved, modify it in-memory for the frontend before returning
            if message_obj:
                # The frontend expects the rich content in the 'content' field.
                # The DB has the rich content in metadata.frontend_content.
                # Let's reconstruct the message for yielding.
                message_for_yield = message_obj.copy()
                message_for_yield["content"] = structured_result_for_frontend
                return message_for_yield

            return message_obj  # Return the modified message object
        except Exception as e:
            logger.error(f"Error adding tool result: {str(e)}", exc_info=e)
            # Fallback to a simple message
            try:
                fallback_message = {"role": "user", "content": str(result)}
                message_obj = await self.add_message(
                    thread_id=thread_id,
                    type="tool",
                    content=fallback_message,
                    is_llm_message=True,
                    metadata={"assistant_message_id": assistant_message_id} if assistant_message_id else {},
                )
                return message_obj  # Return the full message object
            except Exception as e2:
                logger.error(f"Failed even with fallback message: {str(e2)}", exc_info=e2)
                return None  # Return None on error

    @staticmethod
    def _create_structured_tool_result(
        tool_call: dict[str, Any],
        result: ToolResult,
        parsing_details: Optional[dict[str, Any]] = None,
        for_llm: bool = False,
    ):
        """Create a structured tool result format that's tool-agnostic and provides rich information.

        Args:
            tool_call: The original tool call that was executed
            result: The result from the tool execution
            parsing_details: Optional parsing details for XML calls
            for_llm: If True, creates a concise version for the LLM context.

        Returns:
            Structured dictionary containing tool execution information
        """
        # Extract tool information
        function_name = tool_call.get("function_name", "unknown")
        xml_tag_name = tool_call.get("xml_tag_name")
        arguments = tool_call.get("arguments", {})
        tool_call_id = tool_call.get("id")

        # Process the output - if it's a JSON string, parse it back to an object
        output = result.output if hasattr(result, "output") else str(result)
        if isinstance(output, str):
            try:
                # Try to parse as JSON to provide structured data to frontend
                parsed_output = safe_json_parse(output)
                # If parsing succeeded and we got a dict/list, use the parsed version
                if isinstance(parsed_output, (dict, list)):
                    output = parsed_output
                # Otherwise keep the original string
            except Exception:
                # If parsing fails, keep the original string
                pass

        output_to_use = output
        # If this is for the LLM and it's an edit_file tool, create a concise output
        if for_llm and function_name == "edit_file" and isinstance(output, dict):
            # The frontend needs original_content and updated_content to render diffs.
            # The concise version for the LLM was causing issues.
            # We will now pass the full output, and rely on the ContextManager to truncate if needed.
            output_to_use = output

        # Create the structured result
        structured_result_v1 = {
            "tool_execution": {
                "function_name": function_name,
                "xml_tag_name": xml_tag_name,
                "tool_call_id": tool_call_id,
                "arguments": arguments,
                "result": {
                    "success": result.success if hasattr(result, "success") else True,
                    "output": output_to_use,  # This will be either rich or concise based on `for_llm`
                    "error": getattr(result, "error", None) if hasattr(result, "error") else None,
                },
            }
        }

        return structured_result_v1

    @staticmethod
    def _create_tool_context(
        tool_call: dict[str, Any],
        tool_index: int,
        assistant_message_id: Optional[str] = None,
        parsing_details: Optional[dict[str, Any]] = None,
    ) -> ToolExecutionContext:
        """Create a tool execution context with display name and parsing details populated."""
        context = ToolExecutionContext(
            tool_call=tool_call,
            tool_index=tool_index,
            assistant_message_id=assistant_message_id,
            parsing_details=parsing_details,
        )

        # Set function_name and xml_tag_name fields
        if "xml_tag_name" in tool_call:
            context.xml_tag_name = tool_call["xml_tag_name"]
            context.function_name = tool_call.get("function_name", tool_call["xml_tag_name"])
        else:
            # For non-XML tools, use function name directly
            context.function_name = tool_call.get("function_name", "unknown")
            context.xml_tag_name = None

        return context

    async def _yield_and_save_tool_started(
        self, context: ToolExecutionContext, thread_id: str, thread_run_id: str
    ) -> Optional[dict[str, Any]]:
        """Formats, saves, and returns a tool started status message."""
        tool_name = context.xml_tag_name or context.function_name
        content = {
            "role": "assistant",
            "status_type": "tool_started",
            "function_name": context.function_name,
            "xml_tag_name": context.xml_tag_name,
            "message": f"Starting execution of {tool_name}",
            "tool_index": context.tool_index,
            "tool_call_id": context.tool_call.get("id"),  # Include tool_call ID if native
        }
        metadata = {"thread_run_id": thread_run_id}
        saved_message_obj = await self.add_message(
            thread_id=thread_id, type="status", content=content, is_llm_message=False, metadata=metadata
        )
        return saved_message_obj  # Return the full object (or None if saving failed)

    async def _yield_and_save_tool_completed(
        self, context: ToolExecutionContext, tool_message_id: Optional[str], thread_id: str, thread_run_id: str
    ) -> Optional[dict[str, Any]]:
        """Formats, saves, and returns a tool completed/failed status message."""
        if not context.result:
            # Delegate to error saving if result is missing (e.g., execution failed)
            return await self._yield_and_save_tool_error(context, thread_id, thread_run_id)

        tool_name = context.xml_tag_name or context.function_name
        status_type = "tool_completed" if context.result.success else "tool_failed"
        message_text = f"Tool {tool_name} {'completed successfully' if context.result.success else 'failed'}"

        content = {
            "role": "assistant",
            "status_type": status_type,
            "function_name": context.function_name,
            "xml_tag_name": context.xml_tag_name,
            "message": message_text,
            "tool_index": context.tool_index,
            "tool_call_id": context.tool_call.get("id"),
        }
        metadata = {"thread_run_id": thread_run_id}
        # Add the *actual* tool result message ID to the metadata if available and successful
        if context.result.success and tool_message_id:
            metadata["linked_tool_result_message_id"] = tool_message_id

        # <<< ADDED: Signal if this is a terminating tool >>>
        if context.function_name in ["ask", "complete", "present_presentation"]:
            metadata["agent_should_terminate"] = "true"
            logger.debug(f"Marking tool status for '{context.function_name}' with termination signal.")
        # <<< END ADDED >>>

        saved_message_obj = await self.add_message(
            thread_id=thread_id, type="status", content=content, is_llm_message=False, metadata=metadata
        )
        return saved_message_obj

    async def _yield_and_save_tool_error(
        self, context: ToolExecutionContext, thread_id: str, thread_run_id: str
    ) -> Optional[dict[str, Any]]:
        """Formats, saves, and returns a tool error status message."""
        error_msg = str(context.error) if context.error else "Unknown error during tool execution"
        tool_name = context.xml_tag_name or context.function_name
        content = {
            "role": "assistant",
            "status_type": "tool_error",
            "function_name": context.function_name,
            "xml_tag_name": context.xml_tag_name,
            "message": f"Error executing tool {tool_name}: {error_msg}",
            "tool_index": context.tool_index,
            "tool_call_id": context.tool_call.get("id"),
        }
        metadata = {"thread_run_id": thread_run_id}
        # Save the status message with is_llm_message=False
        saved_message_obj = await self.add_message(
            thread_id=thread_id, type="status", content=content, is_llm_message=False, metadata=metadata
        )
        return saved_message_obj
