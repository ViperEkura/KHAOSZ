"""Unit tests for GenerateResult accumulator and InferenceEngine.generate()."""

import threading
from unittest.mock import MagicMock, patch

from astrai.inference.engine import GenerateResult
from astrai.inference.task import STOP


def test_result_append_single():
    r = GenerateResult(count=1)
    r.append("hello", 0)
    assert r.results[0] == "hello"


def test_result_append_multiple_tasks():
    r = GenerateResult(count=3)
    r.append("a", 0)
    r.append("b", 1)
    r.append("c", 2)
    assert r.results[0] == "a"
    assert r.results[1] == "b"
    assert r.results[2] == "c"


def test_result_stop_marks_complete():
    r = GenerateResult(count=2)
    r.append("text", 0)
    r.append(STOP, 0)
    r.append("more", 1)
    assert r._done[0] is True
    assert r._done[1] is False
    assert r._completed == 1


def test_result_stop_does_not_double_count():
    r = GenerateResult(count=1)
    r.append(STOP, 0)
    r.append(STOP, 0)
    assert r._completed == 1


def test_result_pop_all_returns_and_clears():
    r = GenerateResult(count=2)
    r.append("a", 0)
    r.append("b", 1)
    out = r.pop_all()
    assert len(out) == 2
    assert out[0] == (0, "a")
    assert out[1] == (1, "b")
    assert r.pop_all() == []


def test_result_wait_blocks_until_data():
    r = GenerateResult(count=1)

    def delayed_append():
        import time

        time.sleep(0.05)
        r.append("delayed", 0)

    t = threading.Thread(target=delayed_append)
    t.start()
    ok = r.wait(timeout=5.0)
    t.join()
    assert ok
    assert r.results[0] == "delayed"


def test_result_wait_timeout():
    r = GenerateResult(count=1)
    ok = r.wait(timeout=0.01)
    assert not ok


def test_result_wait_completion_non_streaming():
    r = GenerateResult(count=2)

    def finish_later():
        import time

        time.sleep(0.05)
        r.append(STOP, 0)
        time.sleep(0.05)
        r.append(STOP, 1)

    t = threading.Thread(target=finish_later)
    t.start()
    r.wait_completion()
    t.join()
    assert r._completed == 2


def test_result_get_results():
    r = GenerateResult(count=2)
    r.append("hello", 0)
    r.append("world", 1)
    results = r.get_results()
    assert results == ["hello", "world"]


def test_engine_generate_non_streaming_single():
    from astrai.inference.engine import InferenceEngine

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.decode.return_value = "response"
    mock_tokenizer.stop_ids = [0]

    with patch("astrai.inference.engine.InferenceScheduler") as MockSched:
        instance = MockSched.return_value

        def fake_add(prompt, **kw):
            cb = kw["stream_callback"]
            cb("response")
            cb(STOP)

        instance.add_task.side_effect = fake_add
        instance.remove_task.return_value = []

        eng = InferenceEngine(mock_model, mock_tokenizer, max_batch_size=1)
        result = eng.generate("hello")
        assert result == "response"


def test_engine_generate_streaming_yields_tokens():
    from astrai.inference.engine import InferenceEngine

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.decode.return_value = "tok"
    mock_tokenizer.stop_ids = [0]

    callbacks_saved = []

    def capture_cb(prompt, **kw):
        callbacks_saved.append(kw.get("stream_callback"))

    with patch("astrai.inference.engine.InferenceScheduler") as MockSched:
        instance = MockSched.return_value
        instance.add_task.side_effect = capture_cb
        instance.remove_task.return_value = []

        eng = InferenceEngine(mock_model, mock_tokenizer, max_batch_size=1)
        gen = eng.generate("hello", stream=True)

        cb = callbacks_saved[0]
        cb("t1")
        cb("t2")
        cb(STOP)

        tokens = list(gen)
        assert tokens == ["t1", "t2"]


def test_engine_generate_non_streaming_batch():
    from astrai.inference.engine import InferenceEngine

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.decode.return_value = "r"
    mock_tokenizer.stop_ids = [0]

    with patch("astrai.inference.engine.InferenceScheduler") as MockSched:
        instance = MockSched.return_value

        def fake_add(prompt, **kw):
            cb = kw["stream_callback"]
            cb("r")
            cb(STOP)

        instance.add_task.side_effect = fake_add
        instance.remove_task.return_value = []

        eng = InferenceEngine(mock_model, mock_tokenizer, max_batch_size=2)
        results = eng.generate(["hello", "world"])
        assert results == ["r", "r"]
