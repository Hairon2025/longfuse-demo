from langchain_deepseek import ChatDeepSeek

from langfuse_demo.config import Settings


def create_chat_model(settings: Settings) -> ChatDeepSeek:
    """根据应用配置创建 DeepSeek 聊天模型。"""

    return ChatDeepSeek(
        api_key=settings.deepseek_api_key,
        model=settings.deepseek_model,
        temperature=0,
    )