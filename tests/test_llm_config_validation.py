import pytest
from pydantic import ValidationError

from llm_facade.llm_config import LLMConfig


def test_llm_config_required_fields() -> None:
    """Test that LLMConfig properly enforces required fields."""
    # All required fields
    config = LLMConfig(
        openai_api_key="test-key",
        openai_api_base_url="https://api.example.com/v1",
        llm_model="test-model",
    )
    assert config.openai_api_key == "test-key"
    assert config.openai_api_base_url == "https://api.example.com/v1"
    assert config.llm_model == "test-model"

    # Missing openai_api_key
    with pytest.raises(ValidationError):
        LLMConfig(
            openai_api_base_url="https://api.example.com/v1",
            llm_model="test-model",
        )

    # Missing openai_api_base_url
    with pytest.raises(ValidationError):
        LLMConfig(
            openai_api_key="test-key",
            llm_model="test-model",
        )

    # Missing llm_model
    with pytest.raises(ValidationError):
        LLMConfig(
            openai_api_key="test-key",
            openai_api_base_url="https://api.example.com/v1",
        )


def test_llm_config_empty_values() -> None:
    """Test that LLMConfig validates non-empty values."""
    # Empty strings should fail validation
    with pytest.raises(ValidationError):
        LLMConfig(
            openai_api_key="",
            openai_api_base_url="https://api.example.com/v1",
            llm_model="test-model",
        )

    with pytest.raises(ValidationError):
        LLMConfig(
            openai_api_key="test-key",
            openai_api_base_url="",
            llm_model="test-model",
        )

    with pytest.raises(ValidationError):
        LLMConfig(
            openai_api_key="test-key",
            openai_api_base_url="https://api.example.com/v1",
            llm_model="",
        )


def test_llm_config_url_validation() -> None:
    """Test that LLMConfig validates URL format."""
    # Invalid URL format should fail
    with pytest.raises(ValidationError):
        LLMConfig(
            openai_api_key="test-key",
            openai_api_base_url="not-a-url",
            llm_model="test-model",
        )

    # URLs must start with http:// or https://
    with pytest.raises(ValidationError):
        LLMConfig(
            openai_api_key="test-key",
            openai_api_base_url="ftp://api.example.com/v1",
            llm_model="test-model",
        )

    # Valid URLs
    config1 = LLMConfig(
        openai_api_key="test-key",
        openai_api_base_url="http://api.example.com/v1",
        llm_model="test-model",
    )
    assert config1.openai_api_base_url == "http://api.example.com/v1"

    config2 = LLMConfig(
        openai_api_key="test-key",
        openai_api_base_url="https://api.example.com/v1",
        llm_model="test-model",
    )
    assert config2.openai_api_base_url == "https://api.example.com/v1"
