from langfuse_demo.graph import build_graph
from langfuse_demo.schemas import Citation, ResearchState


def main() -> None:
    """运行命令行版研究工作流（Demo）"""

    question: str = input("请输入问题：").strip()

    if not question:
        print("问题不能为空")
        return

    graph = build_graph()

    initial_state: ResearchState = { "question": question,}

    result: ResearchState = graph.invoke(initial_state)

    answer: str | None = result.get("answer")
    citations: list[Citation] | None = result.get("citations")


    if answer is None or citations is None:
        raise RuntimeError("工作流执行完成，但没有生成回答或引用。")


    print("\n回答：")
    print(answer)

    print("\n引用：")
    for citation in citations:
        print(f"- {citation['url']}")
        print(f"  {citation['quote']}")


if __name__ == "__main__":
    main()