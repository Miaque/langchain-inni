import unittest

import spacy
import zh_core_web_lg
from langchain.retrievers import MultiQueryRetriever
from langchain_text_splitters import SpacyTextSplitter
from loguru import logger

from main import vector_store, llm


class LLMTest(unittest.TestCase):
    def test_spacy(self):
        text_splitter = SpacyTextSplitter(pipeline="zh_core_web_lg")
        docs = text_splitter.split_text("通过下载并运行 get-pip.py，你可以为 Python 环境快速安装或修复 pip。完成安装后，建议验证 pip 版本并升级到最新版。如果遇到问题，请提供你的操作系统、Python 版本（python --version）、错误信息，我可以进一步帮你排查！")
        print(docs)

    def test_install(self):
        nlp = spacy.load("zh_core_web_lg")
        nlp = zh_core_web_lg.load()
        docs = nlp("安装过程会显示进度信息，如下载链接和安装状态")
        print(docs)

    def test_retriever(self):
        retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vector_store.as_retriever(), llm=llm)
        unique_docs = retriever_from_llm.invoke("文档的内容")
        print(unique_docs)

    def test_logger(self):
        logger.debug("That's it, beautiful and simple logging!")


if __name__ == '__main__':
    unittest.main()
