from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.llms import CompletionResponse, LLMMetadata

from llm_facade.gemma3 import GemaVllm
from llm_facade.llm_config import LLMConfig


@pytest.fixture
def llm_config() -> LLMConfig:
    """Create a test LLMConfig."""
    return LLMConfig(
        openai_api_key="test-key",
        openai_api_base_url="https://api.example.com/v1",
        llm_model="gemma-3-test-model",
    )


@pytest.fixture
def mock_openai() -> MagicMock:
    """Create a mock OpenAI client."""
    mock = MagicMock()

    # Configure the chat completions
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Test response"
    mock_choice.message = mock_message
    mock_choice.model_dump_json.return_value = '{"message": {"content": "Test response"}}'
    mock_completion.choices = [mock_choice]

    mock.chat.completions.create.return_value = mock_completion

    # Configure the streaming completions
    mock_stream_chunk1 = MagicMock()
    mock_stream_choice1 = MagicMock()
    mock_stream_choice1.delta.content = "Hello"
    mock_stream_chunk1.choices = [mock_stream_choice1]

    mock_stream_chunk2 = MagicMock()
    mock_stream_choice2 = MagicMock()
    mock_stream_choice2.delta.content = " World"
    mock_stream_chunk2.choices = [mock_stream_choice2]

    mock.completions.create.return_value = [mock_stream_chunk1, mock_stream_chunk2]

    return mock


def test_gemma_vllm_initialization(llm_config: LLMConfig) -> None:
    """Test GemaVllm initialization."""
    # Create separate patches that won't affect the GemaVllm class creation
    # but will allow us to verify if OpenAI was called correctly
    with patch("llm_facade.gemma3.print") as mock_print, \
         patch("llm_facade.gemma3.OpenAI", autospec=True) as mock_openai_class:
        # We need to return a real instance or mock that passes isinstance check
        from openai import OpenAI
        real_instance_or_mock = OpenAI(
            api_key=llm_config.openai_api_key,
            base_url=llm_config.openai_api_base_url
        )
        mock_openai_class.return_value = real_instance_or_mock
        
        # Now we can create the instance
        vllm = GemaVllm(config=llm_config)

        # Check that OpenAI was initialized correctly
        mock_openai_class.assert_called_once_with(
            api_key=llm_config.openai_api_key,
            base_url=llm_config.openai_api_base_url,
        )

        # Check that print was called with the initialization message
        mock_print.assert_called_once_with(f"VLLM client initialized {llm_config.openai_api_base_url}")

        # Check that the config is set correctly
        assert vllm.config == llm_config


def test_gemma_vllm_metadata(llm_config: LLMConfig) -> None:
    """Test GemaVllm metadata property."""
    with patch("llm_facade.gemma3.OpenAI"), patch("llm_facade.gemma3.print"):
        vllm = GemaVllm(config=llm_config)
        metadata = vllm.metadata

        assert isinstance(metadata, LLMMetadata)
        assert metadata.model_name == llm_config.llm_model


@patch("llm_facade.gemma3.OpenAI")
def test_gemma_vllm_complete(mock_openai_class: MagicMock, llm_config: LLMConfig, mock_openai: MagicMock) -> None:
    """Test GemaVllm complete method."""
    mock_openai_class.return_value = mock_openai

    with patch("llm_facade.gemma3.print"):
        vllm = GemaVllm(config=llm_config)
        response = vllm.complete("Test prompt")

        # Check that the OpenAI client was called correctly
        mock_openai.chat.completions.create.assert_called_once_with(
            model=llm_config.llm_model,
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.1,
        )

        # Check response
        assert isinstance(response, CompletionResponse)
        assert response.text == "Test response"
        assert vllm.last_log == '{"message": {"content": "Test response"}}'


@patch("llm_facade.gemma3.OpenAI")
def test_gemma_vllm_stream_complete(
    mock_openai_class: MagicMock, llm_config: LLMConfig, mock_openai: MagicMock
) -> None:
    """Test GemaVllm stream_complete method."""
    mock_openai_class.return_value = mock_openai

    with patch("llm_facade.gemma3.print"):
        vllm = GemaVllm(config=llm_config)
        responses = list(vllm.stream_complete("Test prompt"))

        # Check that the OpenAI client was called correctly
        mock_openai.completions.create.assert_called_once_with(
            model=llm_config.llm_model,
            prompt="Test prompt",
            max_tokens=100,
            temperature=0.1,
            stream=True,
        )

        # Check responses
        assert len(responses) == 2
        assert responses[0].delta == "Hello"
        assert responses[0].text == "Hello"
        assert responses[1].delta == " World"
        assert responses[1].text == "Hello World"
