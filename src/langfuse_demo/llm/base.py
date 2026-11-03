from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BaseAgent(
    ABC,
    Generic[InputT, OutputT],
):
    """LLM Agent 的通用抽象基类。"""

    def __init__(
        self,
        model: BaseChatModel,
    ) -> None:
        self._model: BaseChatModel = model

    @abstractmethod
    def invoke(
        self,
        agent_input: InputT,
    ) -> OutputT:
        """执行 LLM 请求并返回结果。"""

        raise NotImplementedError