from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.llms import CompletionResponse, LLMMetadata

from llm_facade.llm_config import LLMConfig
from llm_facade.qwen3 import QwenVllm


@pytest.fixture
def llm_config() -> LLMConfig:
    """Create a test LLMConfig."""
    return LLMConfig(
        openai_api_key="test-key",
        openai_api_base_url="https://api.example.com/v1",
        llm_model="qwen3-test-model",
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
    mock_choice.finish_reason = "stop"
    mock_choice.model_dump_json.return_value = '{"message": {"content": "Test response"}, "finish_reason": "stop"}'
    mock_completion.choices = [mock_choice]

    mock.chat.completions.create.return_value = mock_completion

    # Configure the streaming completions
    mock_stream_chunk1 = MagicMock()
    mock_stream_choice1 = MagicMock()
    mock_stream_choice1.delta.content = "Hello"
    mock_stream_choice1.delta.tool_calls = None
    mock_stream_chunk1.choices = [mock_stream_choice1]

    mock_stream_chunk2 = MagicMock()
    mock_stream_choice2 = MagicMock()
    mock_stream_choice2.delta.content = " World"
    mock_stream_choice2.delta.tool_calls = None
    mock_stream_chunk2.choices = [mock_stream_choice2]

    # Tool call mock
    mock_stream_chunk3 = MagicMock()
    mock_stream_choice3 = MagicMock()
    mock_stream_choice3.delta.content = None
    mock_stream_choice3.delta.tool_calls = [{"type": "function", "function": {"name": "test_tool"}}]
    mock_stream_chunk3.choices = [mock_stream_choice3]
    mock_stream_chunk3.model_dump_json.return_value = '{"choices": [{"delta": {"tool_calls": [{"type": "function"}]}}]}'

    mock.chat.completions.create.return_value = [mock_stream_chunk1, mock_stream_chunk2, mock_stream_chunk3]

    return mock


@patch("llm_facade.qwen3.OpenAI")
def test_qwen_vllm_initialization(mock_openai_class: MagicMock, llm_config: LLMConfig) -> None:
    """Test QwenVllm initialization."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    vllm = QwenVllm(config=llm_config)

    # Check that OpenAI was initialized correctly
    mock_openai_class.assert_called_once_with(
        api_key=llm_config.openai_api_key,
        base_url=llm_config.openai_api_base_url,
    )

    # Check that the client is set
    assert vllm.client == mock_client
    assert vllm.config == llm_config
    assert vllm.last_log == ""


def test_qwen_vllm_metadata(llm_config: LLMConfig) -> None:
    """Test QwenVllm metadata property."""
    with patch("llm_facade.qwen3.OpenAI"):
        vllm = QwenVllm(config=llm_config)
        metadata = vllm.metadata

        assert isinstance(metadata, LLMMetadata)
        assert metadata.model_name == llm_config.llm_model
        assert metadata.is_chat_model is True
        assert metadata.is_function_calling_model is False


@patch("llm_facade.qwen3.OpenAI")
def test_qwen_vllm_complete(mock_openai_class: MagicMock, llm_config: LLMConfig, mock_openai: MagicMock) -> None:
    """Test QwenVllm complete method."""
    mock_openai_class.return_value = mock_openai
    mock_logger = MagicMock()

    vllm = QwenVllm(config=llm_config, logger=mock_logger)
    response = vllm.complete("Test prompt")

    # Check that the OpenAI client was called correctly
    mock_openai.chat.completions.create.assert_called_once_with(
        model=llm_config.llm_model,
        messages=[{"role": "user", "content": "Test prompt /no_think"}],
        presence_penalty=1.5,
        top_p=0.8,
        temperature=0.7,
        extra_body={"top_k": 20},
    )

    # Check response
    assert isinstance(response, CompletionResponse)
    assert response.text == "Test response"
    assert vllm.last_log == '{"message": {"content": "Test response"}, "finish_reason": "stop"}'


@patch("llm_facade.qwen3.OpenAI")
def test_qwen_vllm_complete_with_none_content(mock_openai_class: MagicMock, llm_config: LLMConfig) -> None:
    """Test QwenVllm complete method when message content is None."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    # Configure the chat completions with None content
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = None  # Content is None
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    # Mock model_dump_json to raise an exception
    mock_choice.model_dump_json.side_effect = Exception("Error in model_dump_json")

    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion

    mock_logger = MagicMock()
    vllm = QwenVllm(config=llm_config, logger=mock_logger)
    response = vllm.complete("Test prompt")

    # Check response with None content
    assert response.text == ""

    # Check that logger.exception was called
    mock_logger.exception.assert_called_once_with("Error in model_dump_json")


@patch("llm_facade.qwen3.OpenAI")
def test_qwen_vllm_complete_with_length_limit(mock_openai_class: MagicMock, llm_config: LLMConfig) -> None:
    """Test QwenVllm complete method when finish_reason is length."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    # Configure the chat completions
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Truncated response"
    mock_choice.message = mock_message
    mock_choice.finish_reason = "length"  # Length limit reached
    mock_choice.model_dump_json.return_value = '{"message": {"content": "Truncated response"}}'
    mock_completion.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_completion

    mock_logger = MagicMock()
    vllm = QwenVllm(config=llm_config, logger=mock_logger)
    response = vllm.complete("Test prompt")

    # Check response
    assert response.text == "Truncated response"

    # Check that warning was logged
    mock_logger.warning.assert_called_once_with("Completion stopped due to length limit.")


@patch("llm_facade.qwen3.OpenAI")
def test_qwen_vllm_stream_complete(mock_openai_class: MagicMock, llm_config: LLMConfig, mock_openai: MagicMock) -> None:
    """Test QwenVllm stream_complete method."""
    mock_openai_class.return_value = mock_openai

    vllm = QwenVllm(config=llm_config)
    responses = list(vllm.stream_complete("Test prompt"))

    # Check that the OpenAI client was called correctly
    mock_openai.chat.completions.create.assert_called_once_with(
        model=llm_config.llm_model,
        messages=[{"role": "user", "content": "Test prompt /no_think"}],
        presence_penalty=1.5,
        top_p=0.8,
        temperature=0.7,
        extra_body={"top_k": 20},
        stream=True,
    )

    # Check responses
    assert len(responses) == 2
    assert responses[0].delta == "Hello"
    assert responses[0].text == "Hello"
    assert responses[1].delta == " World"
    assert responses[1].text == "Hello World"

    # Check that last_log was set with the tool call
    assert (
        vllm.last_log == 'Tool call received in chunk: {"choices": [{"delta": {"tool_calls": [{"type": "function"}]}}]}'
    )


@patch("llm_facade.qwen3.OpenAI")
def test_qwen_vllm_stream_complete_with_special_char_replacement(
    mock_openai_class: MagicMock, llm_config: LLMConfig
) -> None:
    """Test QwenVllm stream_complete method with special character replacement."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    # Configure the streaming completions with special character
    mock_stream_chunk = MagicMock()
    mock_stream_choice = MagicMock()
    mock_stream_choice.delta.content = "ß is replaced"  # Contains ß
    mock_stream_choice.delta.tool_calls = None
    mock_stream_chunk.choices = [mock_stream_choice]
    mock_client.chat.completions.create.return_value = [mock_stream_chunk]

    vllm = QwenVllm(config=llm_config)
    responses = list(vllm.stream_complete("Test prompt"))

    # Check response with replaced ß
    assert responses[0].delta == "ss is replaced"
    assert responses[0].text == "ss is replaced"
