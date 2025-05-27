from llm_facade.llm_config import LLMConfig
from llm_facade.qwen3 import QwenVllm
from structlog import get_logger


def test_ctor() -> None:
    config = LLMConfig(
        openai_api_key="test-key",
        openai_api_base_url="https://api.example.com/v1",
        llm_model="test-model",
    )

    logger = get_logger()

    llm = QwenVllm(config=config, logger=logger)
