import json
from typing import cast

from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from langfuse_demo.llm.base import BaseAgent
from langfuse_demo.schemas import AnswerGenerationInput, AnswerGenerationResult


ANSWER_GENERATION_PROMPT: str = """
你是一个严格依据检索资料回答问题的助手。

要求：
1. 只能使用提供的 sources，不得补充资料中不存在的事实。
2. 如果资料不足，要在 answer 中明确说明。
3. citations 中的 url 必须来自 sources。
4. citations 中的 quote 必须逐字复制对应 source 的 content 片段。
5. 只引用回答中实际使用过的来源。
6. 请严格按照结构化输出格式返回结果，不要添加额外字段。
""".strip()


class AnswerGeneratorAgent(
    BaseAgent[AnswerGenerationInput, AnswerGenerationResult],
):
    """依据检索来源生成回答和引用的 LLM Agent。"""

    def __init__(self, model: BaseChatModel) -> None:
        super().__init__(model)
        self._structured_model: Runnable[
            LanguageModelInput,
            AnswerGenerationResult,
        ] = cast(
            Runnable[LanguageModelInput, AnswerGenerationResult],
            self._model.with_structured_output(AnswerGenerationResult),
        )

    def invoke(
        self,
        agent_input: AnswerGenerationInput,
    ) -> AnswerGenerationResult:
        """根据问题和固定检索来源生成回答。"""

        question: str = agent_input.question.strip()
        if not question:
            raise ValueError("问题不能为空。")
        if not agent_input.sources:
            raise ValueError("sources 不能为空。")

        sources_json: str = json.dumps(
            agent_input.sources,
            ensure_ascii=False,
            indent=2,
        )
        prompt: str = (
            f"{ANSWER_GENERATION_PROMPT}\n\n"
            f"问题分类：{agent_input.category}\n"
            f"用户问题：{question}\n\n"
            f"sources：\n{sources_json}"
        )
        result: AnswerGenerationResult = self._structured_model.invoke(prompt)
        answer: str = result.answer.strip()
        if not answer:
            raise ValueError("LLM 生成的 answer 不能为空。")

        return AnswerGenerationResult(
            answer=answer,
            citations=result.citations,
        )
