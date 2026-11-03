from typing import Callable

from langfuse_demo.llm.base import BaseAgent
from langfuse_demo.schemas import (
    AnswerGenerationInput,
    AnswerGenerationResult,
    Citation,
    ClassificationResult,
    ClassificationUpdate,
    QuestionCategory,
    ResearchState,
    SearchUpdate,
    Source,
    SummaryUpdate,
)


def create_classify_question_node(
    classifier: BaseAgent[str, ClassificationResult],
) -> Callable[[ResearchState], ClassificationUpdate]:
    """
    创建问题分类节点

    分类结果：
        - concept:         概念解释类问题
        - how_to:          操作步骤类问题
        - troubleshooting: 故障排查类问题
    """

    def classify_question(
        state: ResearchState,
    ) -> ClassificationUpdate:
        """读取 State，调用分类 Agent，返回状态更新。"""

        question: str = state["question"].strip()
        if not question:
            raise ValueError("问题不能为空。")

        result: ClassificationResult = classifier.invoke(question)

        update: ClassificationUpdate = {
            "category": result.category,
            "search_query": result.search_query,
        }

        return update

    return classify_question


def create_summarize_answer_node(
    answer_generator: BaseAgent[
        AnswerGenerationInput,
        AnswerGenerationResult,
    ],
) -> Callable[[ResearchState], SummaryUpdate]:
    """创建答案生成节点。"""

    def summarize_answer(
        state: ResearchState,
    ) -> SummaryUpdate:
        """读取 State，调用生成 Agent，返回回答和引用。"""

        question: str = state["question"].strip()
        category: QuestionCategory | None = state.get("category")
        sources: list[Source] | None = state.get("sources")

        if not question:
            raise ValueError("问题不能为空。")
        if category is None:
            raise ValueError("category 不存在，无法生成回答。")
        if not sources:
            raise ValueError("sources 不存在，无法生成回答。")

        agent_input: AnswerGenerationInput = AnswerGenerationInput(
            question=question,
            category=category,
            sources=sources,
        )
        result: AnswerGenerationResult = answer_generator.invoke(agent_input)

        citations: list[Citation] = [
            {
                "url": citation.url,
                "quote": citation.quote,
            }
            for citation in result.citations
        ]
        update: SummaryUpdate = {
            "answer": result.answer,
            "citations": citations,
        }
        return update

    return summarize_answer


def search_documents(state: ResearchState) -> SearchUpdate:
    """
    根据搜索查询词进行文档检索，并返回相关来源。

    目前使用简单的模拟数据，后续可接入实际的搜索引擎或知识库。
    """

    search_query: str = state.get("search_query", "").strip()

    if not search_query:
        raise ValueError("搜索查询词不能为空。")

    # 模拟搜索结果
    sources: list[Source] = [
        {
            "title": "LangGraph State",
            "url": "local://langgraph-state",
            "content": (
                "LangGraph State 是工作流节点之间共享的数据结构。"
                "节点可以读取当前状态，并返回部分状态更新。"
            ),
        },
        {
            "title": "LangGraph Nodes",
            "url": "local://langgraph-nodes",
            "content": (
                "LangGraph 节点是接收当前 State 的 Python 函数。"
                "节点完成处理后，不需要返回完整 State，"
                "只需要返回自己负责更新的字段。"
            ),
        },
        {
            "title": "LangGraph Workflow",
            "url": "local://langgraph-workflow",
            "content": (
                "创建 LangGraph 工作流时，先定义 State 和节点函数，"
                "再使用 StateGraph 注册节点并连接边，最后调用 compile() "
                "得到可执行工作流。"
            ),
        },
    ]

    # 生成状态更新
    update: SearchUpdate = {
        "sources": sources,
    }

    return update
