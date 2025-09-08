from typing import Any, Optional, Union

from loguru import logger

from models.messages import Messages
from models.threads import Threads
from response_processor import ResponseProcessor
from tools.base_tool import BaseTool
from tools.tool_registry import ToolRegistry


class ThreadManager:
    def __init__(self, agent_config: Optional[dict] = None):
        self.tool_registry = ToolRegistry()
        self.agent_config = agent_config
        self.response_processor = ResponseProcessor(
            tool_registry=self.tool_registry,
            add_message_callback=self.add_message,
            agent_config=self.agent_config,
        )

    def add_tool(self, tool_class: type[BaseTool], function_names: Optional[list[str]] = None, **kwargs):
        """向ThreadManager添加工具。"""
        self.tool_registry.register_tool(tool_class, function_names, **kwargs)

    async def create_thread(
        self,
        account_id: Optional[str] = None,
        project_id: Optional[str] = None,
        is_public: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """在数据库中创建新线程。

        参数:
            account_id: 线程的可选账户ID。如果为None，则创建孤立线程。
            project_id: 线程的可选项目ID。如果为None，则创建孤立线程。
            is_public: 线程是否应该公开（默认为False）。
            metadata: 用于额外线程上下文的可选元数据字典。

        返回:
            新创建线程的thread_id。

        异常:
            Exception: 如果线程创建失败。
        """
        logger.debug(
            f"正在创建新线程 (account_id: {account_id}, project_id: {project_id}, is_public: {is_public})"
        )

        # 准备线程创建数据
        thread_data = {"is_public": is_public, "metadata": metadata or {}}

        # 仅在提供时添加可选字段
        if account_id:
            thread_data["account_id"] = account_id
        if project_id:
            thread_data["project_id"] = project_id

        try:
            # 插入线程并获取thread_id
            result = Threads.insert(thread_data)

            if result and result.thread_id:
                thread_id = result.thread_id
                logger.debug(f"成功创建线程: {thread_id}")
                return thread_id
            else:
                logger.error(
                    f"线程创建失败或未返回预期的数据结构。结果数据: {result}"
                )
                raise Exception("Failed to create thread: no thread_id returned")

        except Exception as e:
            logger.error("创建线程失败: ", exc_info=e)
            raise Exception(f"Thread creation failed: {str(e)}")

    async def add_message(
        self,
        thread_id: str,
        type: str,
        content: Union[dict[str, Any], list[Any], str],
        is_llm_message: bool = False,
        metadata: Optional[dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        agent_version_id: Optional[str] = None,
    ):
        """向数据库中的线程添加消息。

        参数:
            thread_id: 要添加消息的线程ID。
            type: 消息类型（例如：'text', 'image_url', 'tool_call', 'tool', 'user', 'assistant'）。
            content: 消息内容。可以是字典、列表或字符串。
                     将作为JSONB存储在数据库中。
            is_llm_message: 指示消息是否来自LLM的标志。
                           默认为False（用户消息）。
            metadata: 用于额外消息元数据的可选字典。
                     默认为None，如果为None则存储为空JSONB对象。
            agent_id: 与此消息关联的代理的可选ID。
            agent_version_id: 使用的特定代理版本的可选ID。
        """
        logger.debug(
            f"正在向线程 {thread_id} 添加类型为 '{type}' 的消息 (agent: {agent_id}, version: {agent_version_id})"
        )

        # 准备插入数据
        data_to_insert = {
            "thread_id": thread_id,
            "type": type,
            "content": content,
            "is_llm_message": is_llm_message,
            "metadata": metadata or {},
        }

        # 如果提供了代理信息则添加
        if agent_id:
            data_to_insert["agent_id"] = agent_id
        if agent_version_id:
            data_to_insert["agent_version_id"] = agent_version_id

        try:
            # 插入消息并获取包含id的插入行数据
            saved_message = Messages.insert(data_to_insert)
            logger.debug(f"成功向线程 {thread_id} 添加消息")

            if saved_message and saved_message.message_id:
                # 如果这是assistant_response_end，尝试在超过限制时扣除积分
                if type == "assistant_response_end" and isinstance(content, dict):
                    try:
                        usage = content.get("usage", {}) if isinstance(content, dict) else {}
                        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                        model = content.get("model") if isinstance(content, dict) else None
                        # 获取此线程的account_id，对于个人账户等于user_id
                        thread_row = Threads.get_by_id(thread_id)
                        user_id = thread_row.account_id if thread_row else None
                    except Exception as billing_e:
                        logger.error(
                            f"处理消息 {saved_message.message_id} 的积分使用错误:",
                            exc_info=billing_e,
                        )
                return saved_message.model_dump()
            else:
                logger.error(
                    "插入操作失败或未为线程 {} 返回预期的数据结构。结果数据: {}",
                    thread_id,
                    saved_message,
                )
                return None
        except Exception as e:
            logger.error(f"向线程 {thread_id} 添加消息失败: ", exc_info=e)
            raise
