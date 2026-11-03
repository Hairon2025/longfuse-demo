from dataclasses import dataclass

from langfuse_demo.llm.base import BaseAgent
from langfuse_demo.schemas import (
    AnswerGenerationInput,
    AnswerGenerationResult,
    ClassificationResult,
)


@dataclass(frozen=True, slots=True)
class LLMComponents:
    """工作流使用的所有 LLM 组件。"""

    classifier: BaseAgent[
        str,
        ClassificationResult,
    ]
    answer_generator: BaseAgent[
        AnswerGenerationInput,
        AnswerGenerationResult,
    ]
