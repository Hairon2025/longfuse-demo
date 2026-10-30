from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import Callable, CompiledStateGraph

from langfuse_demo.nodes import (
    create_classify_question_node,
    search_documents,
    summarize_answer,
)
from langfuse_demo.schemas import ClassificationUpdate, ResearchState

def build_graph(
    model: ChatDeepSeek,
) -> CompiledStateGraph[
    ResearchState,
    None,
    ResearchState,
    ResearchState
]:
    """构建并编译研究工作流。"""

    builder: StateGraph[
        ResearchState,
        None, 
        ResearchState, 
        ResearchState
    ] = StateGraph(ResearchState)

    classify_question: Callable[
        [ResearchState],
        ClassificationUpdate,
    ] = create_classify_question_node(model)

    # 添加节点
    builder.add_node("classify_question", classify_question)
    builder.add_node("search_documents", search_documents)
    builder.add_node("summarize_answer", summarize_answer)

    # 添加边
    builder.add_edge(START, "classify_question")
    builder.add_edge("classify_question", "search_documents")
    builder.add_edge("search_documents", "summarize_answer")
    builder.add_edge("summarize_answer", END)

    # 构建图
    graph: CompiledStateGraph[
        ResearchState,
        None,
        ResearchState,
        ResearchState
    ] = builder.compile(name="research-workflow")

    return graph