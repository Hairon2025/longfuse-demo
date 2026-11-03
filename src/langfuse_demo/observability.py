from collections.abc import Generator
from contextlib import contextmanager

from langchain_core.runnables import RunnableConfig
from langfuse import get_client
from langfuse.langchain import CallbackHandler


@contextmanager
def trace_run(
    *,
    tags: list[str] | None = None,
    metadata: dict[str, str] | None = None,
) -> Generator[RunnableConfig, None, None]:
    """创建一次带 Langfuse 追踪的运行上下文。"""

    handler: CallbackHandler = CallbackHandler()

    run_config: RunnableConfig = {
        "callbacks": [handler],
    }

    if tags is not None:
        run_config["tags"] = tags

    if metadata is not None:
        run_config["metadata"] = metadata

    try:
        yield run_config
    finally:
        get_client().flush()