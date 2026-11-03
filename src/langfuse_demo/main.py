from langchain_deepseek import ChatDeepSeek
from langgraph.graph.state import CompiledStateGraph

from langfuse_demo.config import Settings, load_settings
from langfuse_demo.graph import build_graph
from langfuse_demo.llm.answer_generator import AnswerGeneratorAgent
from langfuse_demo.llm.clients import create_chat_model
from langfuse_demo.llm.components import LLMComponents
from langfuse_demo.llm.question_classifier import (
    QuestionClassifierAgent,
)
from langfuse_demo.schemas import Citation, ResearchState
from langfuse_demo.observability import trace_run

def main() -> None:
    """运行命令行版研究工作流。"""

    question: str = input("请输入问题：").strip()

    if not question:
        print("问题不能为空")
        return

    settings: Settings = load_settings()

    model: ChatDeepSeek = create_chat_model(settings)

    classifier: QuestionClassifierAgent = QuestionClassifierAgent(model)
    answer_generator: AnswerGeneratorAgent = AnswerGeneratorAgent(model)

    llm_components: LLMComponents = LLMComponents(
        classifier=classifier,
        answer_generator=answer_generator,
    )

    graph: CompiledStateGraph[
        ResearchState,
        None,
        ResearchState,
        ResearchState,
    ] = build_graph(llm_components)

    initial_state: ResearchState = {
        "question": question,
    }

    with trace_run(
    tags=["langgraph-demo"],
    metadata={
        "workflow": "research-workflow",
        "model": settings.deepseek_model,
    },
    ) as run_config:
        result: ResearchState = graph.invoke(
            initial_state,
            config=run_config,
        )

    answer: str | None = result.get("answer")
    citations: list[Citation] | None = result.get(
        "citations",
    )

    if answer is None or citations is None:
        raise RuntimeError(
            "工作流执行完成，但没有生成回答或引用。",
        )

    print("\n分类：")
    print(result.get("category"))

    print("\n检索词：")
    print(result.get("search_query"))

    print("\n回答：")
    print(answer)

    print("\n引用：")
    for citation in citations:
        print(f"- {citation['url']}")
        print(f"  {citation['quote']}")


if __name__ == "__main__":
    main()
