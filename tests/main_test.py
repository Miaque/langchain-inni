import unittest

from langchain_core.tools import Tool
from langchain_text_splitters import SpacyTextSplitter
from langgraph.checkpoint.postgres import PostgresSaver
from loguru import logger

from configs import app_config, WORK_DIR
from main import tool_registry


class LLMTest(unittest.TestCase):
    def test_spacy(self):
        text_splitter = SpacyTextSplitter(pipeline="zh_core_web_lg")
        docs = text_splitter.split_text(
            "通过下载并运行 get-pip.py，你可以为 Python 环境快速安装或修复 pip。完成安装后，建议验证 pip 版本并升级到最新版。如果遇到问题，请提供你的操作系统、Python 版本（python --version）、错误信息，我可以进一步帮你排查！"
        )
        print(docs)

    def test_logger(self):
        logger.debug("That's it, beautiful and simple logging!")

    def test_core_tools(self):
        functions = tool_registry.get_available_functions()
        # logger.info(functions)

        for name, func in functions.items():
            # Tool(name=,func=,description=)
            desc = tool_registry.get_tool(name)["schema"].schema["function"]["description"]
            params = tool_registry.get_tool(name)["schema"].schema["function"]["parameters"]
            tool = Tool(name=name, func=func, description=desc, args_schema=params)
            logger.info(tool)

    def test_init(self):
        DB_URI = app_config.SQLALCHEMY_DATABASE_URI
        logger.info(DB_URI)

        with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
            checkpointer.setup()

    def test_dir(self):
        logger.info(WORK_DIR)



if __name__ == "__main__":
    unittest.main()
