"""Common test fixtures and utilities for tests."""

from unittest.mock import MagicMock

import pytest
from openai import OpenAI
from structlog.stdlib import BoundLogger


class MockOpenAI(MagicMock):
    """A mock that passes isinstance(obj, OpenAI) checks."""

    def __instancecheck__(self, instance):
        return True

    def __subclasscheck__(self, subclass):
        return True


# Make MockOpenAI look like it's an OpenAI
OpenAI.__instancecheck__ = lambda cls, instance: isinstance(instance, OpenAI | MockOpenAI)


class MockBoundLogger(MagicMock):
    """A mock that passes isinstance(obj, BoundLogger) checks."""

    def __instancecheck__(self, instance):
        return True

    def __subclasscheck__(self, subclass):
        return True


# Make MockBoundLogger look like it's a BoundLogger
BoundLogger.__instancecheck__ = lambda cls, instance: isinstance(instance, BoundLogger | MockBoundLogger)


@pytest.fixture
def mock_openai() -> MockOpenAI:
    """Create a mock OpenAI client that passes isinstance checks."""
    mock = MockOpenAI()

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
    mock_stream_chunk1.choices = [mock_stream_choice1]

    mock_stream_chunk2 = MagicMock()
    mock_stream_choice2 = MagicMock()
    mock_stream_choice2.delta.content = " World"
    mock_stream_chunk2.choices = [mock_stream_choice2]

    mock.completions.create.return_value = [mock_stream_chunk1, mock_stream_chunk2]

    return mock


@pytest.fixture
def mock_logger() -> MockBoundLogger:
    """Create a mock logger that passes isinstance checks."""
    return MockBoundLogger()
