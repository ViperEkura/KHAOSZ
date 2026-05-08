"""Unit tests for the inference HTTP server."""

from unittest.mock import MagicMock

import pytest


def test_health_no_model(client, monkeypatch):
    """GET /health should return 200 even when engine not loaded."""
    monkeypatch.setattr("astrai.inference.server._state.engine", None)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert not data["model_loaded"]


def test_health_with_model(client, loaded_model):
    """GET /health should return 200 when engine is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_generate_non_stream(client, loaded_model, monkeypatch):
    """POST /generate with stream=false should return JSON response."""
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
    assert "response" in data


def test_generate_stream(client, loaded_model, monkeypatch):
    """POST /generate with stream=true should return plain text stream."""

    async def async_gen():
        yield "chunk1"
        yield "chunk2"

    mock_engine = loaded_model
    mock_engine.generate_async.return_value = async_gen()
    monkeypatch.setattr("astrai.inference.server._state.engine", mock_engine)
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
    content = response.content.decode("utf-8")
    assert "chunk1" in content
    assert "chunk2" in content


def test_chat_completions_non_stream(client, loaded_model, monkeypatch):
    """POST /v1/chat/completions with stream=false returns OpenAI-style JSON."""

    async def async_gen():
        yield "Assistant reply"

    mock_engine = loaded_model
    mock_engine.generate_async.return_value = async_gen()
    monkeypatch.setattr("astrai.inference.server._state.engine", mock_engine)
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.8,
            "max_tokens": 100,
            "stream": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert "usage" in data
    assert "prompt_tokens" in data["usage"]


def test_chat_completions_stream(client, loaded_model, monkeypatch):
    """POST /v1/chat/completions with stream=true returns SSE stream."""

    async def async_gen():
        yield "cumulative1"
        yield "cumulative2"

    mock_engine = loaded_model
    mock_engine.generate_async.return_value = async_gen()
    monkeypatch.setattr("astrai.inference.server._state.engine", mock_engine)
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.8,
            "max_tokens": 100,
            "stream": True,
        },
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    lines = [
        line.strip() for line in response.content.decode("utf-8").split("\n") if line
    ]
    assert any("cumulative1" in line for line in lines)
    assert any("cumulative2" in line for line in lines)
    assert any("[DONE]" in line for line in lines)


def test_generate_with_history(client, loaded_model, monkeypatch):
    """POST /generate with history parameter."""
    response = client.post(
        "/generate",
        params={
            "query": "Hi",
            "history": [["user1", "assistant1"], ["user2", "assistant2"]],
            "stream": False,
        },
    )
    assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
