import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
ENV_FILE: Path = PROJECT_ROOT / ".env"

@dataclass(frozen=True, slots=True)
class Settings:
    """应用运行所需的配置。"""

    deepseek_api_key: str = field(repr=False)
    deepseek_model: str = "deepseek-flash"


def load_settings() -> Settings:
    """从项目根目录的 .env 加载并校验配置。"""

    load_dotenv(dotenv_path=ENV_FILE)

    deepseek_api_key: str | None = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise RuntimeError("缺少环境变量 DEEPSEEK_API_KEY")

    deepseek_model: str = os.getenv(
        "DEEPSEEK_MODEL",
        "deepseek-flash",
    )

    return Settings(
        deepseek_api_key=deepseek_api_key,
        deepseek_model=deepseek_model,
    )