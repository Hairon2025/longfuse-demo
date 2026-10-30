from typing import Literal, NotRequired, TypedDict, TypeAlias
from pydantic import BaseModel, ConfigDict, Field

QuestionCategory: TypeAlias = Literal[
    "concept",
    "how_to",
    "troubleshooting",
]

class Source(TypedDict):
    """单条检索来源。"""
    title: str      # 来源标题
    url: str        # 来源链接
    content: str    # 与问题相关的摘要片段

class Citation(TypedDict):
    """回答中使用的一条引用。"""

    url: str        # 引用对应的来源链接
    quote: str      # 从来源内容中引用的原文片段

class ResearchState(TypedDict):
    """
    LangGraph 研究流程的共享状态。

    状态变化：
        question
        -> category、search_query
        -> sources
        -> answer、citations
    """

    # 用户原始问题，由调用方传入，全程保留。
    question: str

    # 问题分类，由 classify_question 节点产生。
    # - concept:         概念解释类问题
    # - how_to:          操作步骤类问题
    # - troubleshooting: 故障排查类问题
    category: NotRequired[QuestionCategory]

    # 实际用于搜索的查询词，由 classify_question 节点产生。
    # 第一版可以直接等于 question，以后可由 LLM 改写。
    search_query: NotRequired[str]

    # 搜索结果，由 search_documents 节点产生。
    sources: NotRequired[list[Source]]

    # 最终回答正文，由 summarize_answer 节点产生。
    answer: NotRequired[str]

    # 回答使用的引用，由 summarize_answer 节点产生。
    citations: NotRequired[list[Citation]]


class ClassificationUpdate(TypedDict):
    """classify_question 节点产生的状态更新。"""

    category: QuestionCategory
    search_query: str

class SearchUpdate(TypedDict):
    """search_documents 节点产生的状态更新。"""

    sources: list[Source]

class SummaryUpdate(TypedDict):
    """summarize_answer 节点产生的状态更新。"""

    answer: str
    citations: list[Citation]

class ClassificationResult(BaseModel):
    """LLM 返回的问题分类结果。"""

    model_config = ConfigDict(extra="forbid")

    category: QuestionCategory = Field(
        description=(
            "问题类型：concept、how_to 或 troubleshooting。"
        ),
    )
    search_query: str = Field(
        min_length=1,
        description="适合用于检索的问题查询词。",
    )
