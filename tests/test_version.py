from unittest.mock import patch

import pytest
from version_pioneer.versionscript import VersionDict

from llm_facade._version import get_version_dict


@pytest.fixture
def mock_version_dict() -> VersionDict:
    """Create a mock version dictionary."""
    return {
        "version": "1.0.0",
        "full-revisionid": "abc123def456",
        "dirty": False,
        "error": None,
        "date": "2023-05-27",
    }


def test_get_version_dict_exists() -> None:
    """Simple test to verify that get_version_dict exists and returns something."""
    result = get_version_dict()
    assert isinstance(result, dict)
    assert "version" in result
    assert isinstance(result["version"], str)


def test_get_version_dict_with_mocks(mock_version_dict: VersionDict) -> None:
    """Test that get_version_dict calls get_version_dict_wo_exec with the expected parameters."""
    with patch("llm_facade._version.get_version_dict_wo_exec") as mock_get:
        mock_get.return_value = mock_version_dict
        
        result = get_version_dict()
        
        # Verify that get_version_dict_wo_exec was called with expected style and prefix
        _, kwargs = mock_get.call_args
        assert kwargs["style"] == "pep440"
        assert kwargs["tag_prefix"] == "v"
        assert result == mock_version_dict
