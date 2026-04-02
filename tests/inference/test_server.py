"""Unit tests for the inference HTTP server."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from astrai.inference.server import app


def test_health_no_model(client, monkeypatch):
    """GET /health should return 200 even when model not loaded."""
    monkeypatch.setattr("astrai.inference.server._model_param", None)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] == False


def test_health_with_model(client, loaded_model):
    """GET /health should return 200 when model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_generate_non_stream(client, loaded_model, mock_generator):
    """POST /generate with stream=false should return JSON response."""
    MockFactory, mock_gen = mock_generator
    mock_gen.generate.return_value = "Test response"
    response = client.post(
        "/generate",
        params={
            "query": "Hello",
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "max_len": 100,
            "stream": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Test response"
    MockFactory.create.assert_called_once()


def test_generate_stream(client, loaded_model, mock_generator):
    """POST /generate with stream=true should return plain text stream."""
    MockFactory, mock_gen = mock_generator
    # Simulate a streaming generator that yields two chunks
    mock_gen.generate.return_value = ["chunk1", "chunk2"]
    response = client.post(
        "/generate",
        params={
            "query": "Hello",
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "max_len": 100,
            "stream": True,
        },
        headers={"Accept": "text/plain"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; charset=utf-8"
    # The stream yields lines ending with newline
    content = response.content.decode("utf-8")
    assert "chunk1" in content
    assert "chunk2" in content


def test_chat_completions_non_stream(client, loaded_model, mock_generator):
    """POST /v1/chat/completions with stream=false returns OpenAI‑style JSON."""
    MockFactory, mock_gen = mock_generator
    mock_gen.generate.return_value = "Assistant reply"
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "max_tokens": 100,
            "stream": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["content"] == "Assistant reply"


def test_chat_completions_stream(client, loaded_model, mock_generator):
    """POST /v1/chat/completions with stream=true returns SSE stream."""
    MockFactory, mock_gen = mock_generator
    # Simulate a streaming generator that yields cumulative responses
    mock_gen.generate.return_value = ["cumulative1", "cumulative2"]
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
            "max_tokens": 100,
            "stream": True,
        },
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    # Parse SSE lines
    lines = [
        line.strip() for line in response.content.decode("utf-8").split("\n") if line
    ]
    # Should contain data lines and a final [DONE]
    assert any("cumulative1" in line for line in lines)
    assert any("cumulative2" in line for line in lines)


def test_generate_with_history(client, loaded_model, mock_generator):
    """POST /generate with history parameter."""
    MockFactory, mock_gen = mock_generator
    mock_gen.generate.return_value = "Response with history"
    response = client.post(
        "/generate",
        params={
            "query": "Hi",
            "history": [["user1", "assistant1"], ["user2", "assistant2"]],
            "stream": False,
        },
    )
    assert response.status_code == 200
    MockFactory.create.assert_called_once()
    # Check that history was passed correctly (currently history is not parsed due to FastAPI limitation)
    call_args = MockFactory.create.call_args
    req = call_args[0][1]  # second argument is GenerationRequest
    # Because history cannot be passed via query params, it will be None
    assert req.history is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
