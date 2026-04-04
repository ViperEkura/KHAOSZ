"""Unit tests for the inference HTTP server."""

import pytest


def test_health_no_model(client, monkeypatch):
    """GET /health should return 200 even when model not loaded."""
    monkeypatch.setattr("astrai.inference.server._model_param", None)
    monkeypatch.setattr("astrai.inference.server._engine", None)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert not data["model_loaded"]
    assert not data["engine_ready"]


def test_health_with_model(client, loaded_model, mock_engine, monkeypatch):
    """GET /health should return 200 when model is loaded."""
    monkeypatch.setattr("astrai.inference.server._engine", mock_engine)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["engine_ready"] is True


def test_generate_non_stream(client, loaded_model, mock_engine, monkeypatch):
    """POST /generate with stream=false should return JSON response."""
    monkeypatch.setattr("astrai.inference.server._engine", mock_engine)
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
    assert data["response"] == "mock response"


def test_generate_stream(client, loaded_model, mock_engine, monkeypatch):
    """POST /generate with stream=true should return plain text stream."""

    # Create a streaming mock
    def stream_gen():
        yield "chunk1"
        yield "chunk2"

    mock_engine.generate.return_value = stream_gen()
    monkeypatch.setattr("astrai.inference.server._engine", mock_engine)
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


def test_chat_completions_non_stream(client, loaded_model, mock_engine, monkeypatch):
    """POST /v1/chat/completions with stream=false returns OpenAI‑style JSON."""
    mock_engine.generate.return_value = "Assistant reply"
    monkeypatch.setattr("astrai.inference.server._engine", mock_engine)
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


def test_chat_completions_stream(client, loaded_model, mock_engine, monkeypatch):
    """POST /v1/chat/completions with stream=true returns SSE stream."""

    # Simulate a streaming generator that yields cumulative responses
    def stream_gen():
        yield "cumulative1"
        yield "cumulative2"
        yield "[DONE]"

    mock_engine.generate.return_value = stream_gen()
    monkeypatch.setattr("astrai.inference.server._engine", mock_engine)
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


def test_generate_with_history(client, loaded_model, mock_engine, monkeypatch):
    """POST /generate with history parameter."""
    monkeypatch.setattr("astrai.inference.server._engine", mock_engine)
    response = client.post(
        "/generate",
        params={
            "query": "Hi",
            "history": [["user1", "assistant1"], ["user2", "assistant2"]],
            "stream": False,
        },
    )
    assert response.status_code == 200
    # Verify the engine.generate was called
    mock_engine.generate.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
