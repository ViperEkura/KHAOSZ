"""Shared fixtures for inference tests."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from astrai.inference.server import app


@pytest.fixture
def client():
    """Provide a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_model_param():
    """Create a mock ModelParameter."""
    mock_param = MagicMock()
    mock_param.model = MagicMock()
    mock_param.tokenizer = MagicMock()
    mock_param.config = MagicMock()
    mock_param.config.max_len = 100
    mock_param.tokenizer.encode = MagicMock(return_value=[1, 2, 3])
    mock_param.tokenizer.decode = MagicMock(return_value="mock response")
    mock_param.tokenizer.stop_ids = []
    mock_param.tokenizer.pad_id = 0
    return mock_param


@pytest.fixture
def mock_generator(mock_model_param):
    """Mock the GeneratorFactory and its generators."""
    with patch("astrai.inference.server.GeneratorFactory") as MockFactory:
        mock_gen = MagicMock()
        mock_gen.generate.return_value = "mock response"
        MockFactory.create.return_value = mock_gen
        yield MockFactory, mock_gen


@pytest.fixture
def loaded_model(mock_model_param, monkeypatch):
    """Simulate that the model is loaded."""
    monkeypatch.setattr("astrai.inference.server._model_param", mock_model_param)
    return mock_model_param
