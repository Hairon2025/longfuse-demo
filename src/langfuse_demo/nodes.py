from langfuse_demo.schemas import (
    Citation,
    ClassificationUpdate,
    QuestionCategory,
    ResearchState,
    SearchUpdate,
    Source,
    SummaryUpdate,
    ClassificationResult,
)
from typing import Callable, cast
from langchain_deepseek import ChatDeepSeek


def create_classify_question_node(model: ChatDeepSeek,) -> Callable[[ResearchState], ClassificationUpdate]:
    """
    对用户问题进行分类，并生成搜索查询词。

    分类结果：
        - concept:         概念解释类问题
        - how_to:          操作步骤类问题
        - troubleshooting: 故障排查类问题
    """
    structured_model = model.with_structured_output(
        ClassificationResult,
    )

    def classify_question(
        state: ResearchState,
    ) -> ClassificationUpdate:
        """使用 LLM 对问题分类，并生成检索查询词。"""

        question: str = state["question"].strip()
        if not question:
            raise ValueError("问题不能为空。")

        prompt: str = f"""
请分析下面的用户问题。

分类规则：
- concept：询问概念、原理或定义
- how_to：询问操作步骤或实现方法
- troubleshooting：出现报错、异常、失败或需要排查问题

同时生成一个简洁、适合检索资料的 search_query。

用户问题：
{question}
""".strip()

        result: ClassificationResult = cast(
            ClassificationResult,
            structured_model.invoke(prompt),
        )

        search_query: str = result.search_query.strip()
        if not search_query:
            raise ValueError("LLM 生成的 search_query 不能为空。")

        update: ClassificationUpdate = {
            "category": result.category,
            "search_query": search_query,
        }

        return update

    return classify_question

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
    ]

    # 生成状态更新
    update: SearchUpdate = {
        "sources": sources,
    }

    return update

def summarize_answer(state: ResearchState) -> SummaryUpdate:
    """根据检索来源生成固定规则的回答和引用。"""

    question: str = state["question"].strip()
    sources: list[Source] | None = state.get("sources")

    if not question:
        raise ValueError("问题不能为空。")

    if not sources:
        raise ValueError("sources 不存在，无法生成回答。")

    source_contents: list[str] = [
        source["content"]
        for source in sources
    ]

    answer: str = (
        f"针对问题“{question}”，检索资料表明："
        f"{''.join(source_contents)}"
    )

    citations: list[Citation] = [
        {
            "url": source["url"],
            "quote": source["content"],
        }
        for source in sources
    ]

    update: SummaryUpdate = {
        "answer": answer,
        "citations": citations,
    }

    return update
