from typing import cast

from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from langfuse_demo.llm.base import BaseAgent
from langfuse_demo.schemas import ClassificationResult


CLASSIFICATION_PROMPT: str = """
请分析下面的用户问题。

分类规则：
- concept：询问概念、原理或定义
- how_to：询问操作步骤或实现方法
- troubleshooting：出现报错、异常、失败或需要排查问题

同时生成一个简洁、适合检索资料的 search_query。

请严格按照结构化输出格式返回结果，不要添加额外字段。

用户问题：
""".strip()


class QuestionClassifierAgent(
    BaseAgent[str, ClassificationResult],
):
    """负责问题分类和检索词生成的 LLM Agent。"""

    def __init__(
        self,
        model: BaseChatModel,
    ) -> None:
        super().__init__(model)

        self._structured_model: Runnable[
            LanguageModelInput,
            ClassificationResult,
        ] = cast(
            Runnable[
                LanguageModelInput,
                ClassificationResult,
            ],
            self._model.with_structured_output(
                ClassificationResult,
            ),
        )

    def invoke(
        self,
        agent_input: str,
    ) -> ClassificationResult:
        """对用户问题进行分类并生成检索词。"""

        question: str = agent_input.strip()
        if not question:
            raise ValueError("问题不能为空。")

        prompt: str = f"{CLASSIFICATION_PROMPT}\n{question}"

        result: ClassificationResult = self._structured_model.invoke(prompt)

        search_query: str = result.search_query.strip()
        if not search_query:
            raise ValueError(
                "LLM 生成的 search_query 不能为空。",
            )

        return ClassificationResult(
            category=result.category,
            search_query=search_query,
        )
