from langfuse_demo.schemas import (
    Citation,
    ClassificationUpdate,
    QuestionCategory,
    ResearchState,
    SearchUpdate,
    Source,
    SummaryUpdate,
)

def classify_question(state: ResearchState) -> ClassificationUpdate:
    """
    对用户问题进行分类，并生成搜索查询词。

    分类结果：
        - concept:         概念解释类问题
        - how_to:          操作步骤类问题
        - troubleshooting: 故障排查类问题
    """

    question: str = state["question"].strip()

    if not question:
        raise ValueError("问题不能为空。")


    # 目前采用简单的关键词匹配方式进行分类，后续可使用 LLM 进行更智能的分类。
    troubleshooting_keywords: tuple[str, ...] = (
        "报错",
        "错误",
        "失败",
        "异常",
        "无法",
        "不工作",
        "不行",
        "不成功",
        "不启动",
        "不显示",
    )

    how_to_keywords: tuple[str, ...] = (
        "怎么",
        "如何",
        "怎样",
        "步骤",
        "方法",
        "操作",
    )

    category: QuestionCategory

    # 关键字匹配分类逻辑：
    if any(keyword in question for keyword in troubleshooting_keywords):
        category = "troubleshooting"
    elif any(keyword in question for keyword in how_to_keywords):
        category = "how_to"
    else:
        category = "concept"

    # 生成状态更新
    update: ClassificationUpdate = {    
        "category": category,
        "search_query": question,  # 目前直接使用原问题作为搜索查询词
    }

    return update

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
