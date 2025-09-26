from langchain_deepseek import ChatDeepSeek

from configs import app_config


def get_instruct_llm():
    return ChatDeepSeek(
        model=app_config.MODEL_NAME,
        api_base=app_config.BASE_URL,
        api_key=app_config.API_KEY,
        temperature=0,
        max_tokens=8192,
    )


def get_reason_llm():
    return ChatDeepSeek(
        model=app_config.REASON_MODEL_NAME,
        api_base=app_config.BASE_URL,
        api_key=app_config.API_KEY,
        temperature=0,
        max_tokens=8192,
    )
