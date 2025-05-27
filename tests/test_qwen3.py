from structlog import get_logger

from llm_facade.llm_config import LLMConfig
from llm_facade.qwen3 import QwenVllm


def test_ctor() -> None:
    config = LLMConfig(
        openai_api_key="test-key",
        openai_api_base_url="https://api.example.com/v1",
        llm_model="test-model",
    )

    logger = get_logger()

    llm = QwenVllm(config=config, logger=logger)

    assert llm.config.openai_api_base_url == config.openai_api_base_url
