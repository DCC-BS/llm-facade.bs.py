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


def test_llm_config_values() -> None:
    """Test that LLMConfig properly handles various values."""
    # Test with valid values
    config1 = LLMConfig(
        openai_api_key="test-key",
        openai_api_base_url="https://api.example.com/v1",
        llm_model="test-model",
    )
    assert config1.openai_api_key == "test-key"
    assert config1.openai_api_base_url == "https://api.example.com/v1"
    assert config1.llm_model == "test-model"

    # Test with empty strings
    # Note: If you want to enforce non-empty strings in LLMConfig,
    # you would need to add constraints in the model definition
    config2 = LLMConfig(
        openai_api_key="",
        openai_api_base_url="https://api.example.com/v1",
        llm_model="test-model",
    )
    assert config2.openai_api_key == ""

    config3 = LLMConfig(
        openai_api_key="test-key",
        openai_api_base_url="",  # Empty URL passes because URL format is not validated
        llm_model="test-model",
    )
    assert config3.openai_api_base_url == ""

    config4 = LLMConfig(
        openai_api_key="test-key",
        openai_api_base_url="https://api.example.com/v1",
        llm_model="",
    )
    assert config4.llm_model == ""


def test_llm_config_url_examples() -> None:
    """Test LLMConfig with different URL formats."""
    # URLs in any format are accepted (URL validation is not enforced in LLMConfig)
    # But if you want to enforce URL validation, you would need to add a validator in the model

    # Regular URL formats
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

    # Non-URL formats are also accepted (but would fail validation if a URL validator were added)
    config3 = LLMConfig(
        openai_api_key="test-key",
        openai_api_base_url="not-a-url",
        llm_model="test-model",
    )
    assert config3.openai_api_base_url == "not-a-url"

    # Other URL schemes are also accepted
    config4 = LLMConfig(
        openai_api_key="test-key",
        openai_api_base_url="ftp://api.example.com/v1",
        llm_model="test-model",
    )
    assert config4.openai_api_base_url == "ftp://api.example.com/v1"
