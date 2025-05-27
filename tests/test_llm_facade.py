from unittest.mock import MagicMock

import pytest
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel

from llm_facade.llm_facade import LLMFacade


class MockResponseModel(BaseModel):
    response: str
    confidence: float


def test_complete() -> None:
    """Test the complete method of LLMFacade."""
    # Create a mock LLM
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Test response"
    mock_llm.complete.return_value = mock_response

    # Initialize LLMFacade with the mock LLM
    facade = LLMFacade(mock_llm)

    # Call the complete method
    result = facade.complete("Test prompt", temperature=0.5)

    # Check that LLM's complete method was called with the right arguments
    mock_llm.complete.assert_called_once_with("Test prompt", temperature=0.5)

    # Check that the result is as expected
    assert result == "Test response"


def test_stream_complete() -> None:
    """Test the stream_complete method of LLMFacade."""
    # Create mock completions
    mock_completions = [
        MagicMock(delta="Hello"),
        MagicMock(delta=" World"),
        MagicMock(delta="!"),
    ]

    # Create a mock LLM
    mock_llm = MagicMock()
    mock_llm.stream_complete.return_value = mock_completions

    # Initialize LLMFacade with the mock LLM
    facade = LLMFacade(mock_llm)

    # Call the stream_complete method
    result = list(facade.stream_complete("Test prompt", temperature=0.5))

    # Check that LLM's stream_complete method was called with the right arguments
    mock_llm.stream_complete.assert_called_once_with("Test prompt", temperature=0.5)

    # Check that the result is as expected
    assert result == ["Hello", " World", "!"]


def test_structured_predict() -> None:
    """Test the structured_predict method of LLMFacade."""
    # Create a mock structured LLM
    mock_structured_llm = MagicMock()
    mock_llm = MagicMock()
    mock_llm.as_structured_llm.return_value = mock_structured_llm

    expected_response = MockResponseModel(response="Test response", confidence=0.9)
    mock_structured_llm.structured_predict.return_value = expected_response

    # Initialize LLMFacade with the mock LLM
    facade = LLMFacade(mock_llm)

    # Create a mock prompt template
    mock_prompt = MagicMock(spec=PromptTemplate)

    # Call the structured_predict method
    result = facade.structured_predict(
        MockResponseModel,
        mock_prompt,
        llm_kwargs={"temperature": 0.5},
        input_text="Test input",
    )

    # Check that LLM's as_structured_llm method was called
    mock_llm.as_structured_llm.assert_called_once()
    
    # Check that structured_llm's structured_predict method was called with the right arguments
    mock_structured_llm.structured_predict.assert_called_once_with(
        MockResponseModel,
        mock_prompt,
        llm_kwargs={"temperature": 0.5},
        input_text="Test input",
    )

    # Check that the result is as expected
    assert result == expected_response
    assert result.response == "Test response"
    assert result.confidence == 0.9
