import pytest
from pydantic import ValidationError

from llm_facade.llm_config import LLMConfig


def test_llm_config_initialization() -> None:
    """Test that LLMConfig can be properly initialized with valid parameters."""
    config = LLMConfig(
        openai_api_key="test-key",
        openai_api_base_url="https://api.example.com/v1",
        llm_model="test-model",
    )

    assert config.openai_api_key == "test-key"
    assert config.openai_api_base_url == "https://api.example.com/v1"
    assert config.llm_model == "test-model"


def test_llm_config_validation() -> None:
    """Test that LLMConfig properly validates input parameters."""
    # Missing required fields should raise ValidationError
    with pytest.raises(ValidationError):
        LLMConfig()  # type: ignore[call-arg]

    # Partial initialization should also raise ValidationError
    with pytest.raises(ValidationError):
        LLMConfig(openai_api_key="test-key")  # type: ignore[call-arg]
