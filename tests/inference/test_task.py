"""Unit tests for Task and TaskManager."""

from unittest.mock import MagicMock

from astrai.inference import STOP, Task, TaskManager, TaskStatus


def _make_mock_tokenizer():
    t = MagicMock()
    t.encode.return_value = [1, 2, 3, 4, 5]
    t.stop_ids = [0]
    return t


def test_task_default_status_is_pending():
    task = Task("id1", [1, 2, 3])
    assert task.status == TaskStatus.PENDING


def test_task_next_pos():
    task = Task("id1", [1, 2, 3])
    task.input_tokens = 5
    assert task.next_pos == 5
    task.output_ids.append(4)
    assert task.next_pos == 6


def test_task_is_finished_max_tokens():
    task = Task("id1", [1, 2, 3], max_tokens=2)
    task.output_tokens = 2
    assert task.is_finished([])


def test_task_is_finished_stop_id():
    task = Task("id1", [1, 2, 3])
    task.output_ids = [5, 0]
    assert task.is_finished([0])


def test_task_is_finished_not_yet():
    task = Task("id1", [1, 2, 3], max_tokens=10)
    task.output_ids = [1, 2]
    assert not task.is_finished([0])


def test_task_manager_add_task():
    tm = TaskManager(tokenizer=_make_mock_tokenizer())
    tid = tm.add_task("hello")
    assert tid.startswith("task_")
    assert tm._total_tasks == 1
    assert len(tm.waiting_queue) == 1


def test_task_manager_add_task_too_long_immediate_stop():
    t = _make_mock_tokenizer()
    t.encode.return_value = list(range(9000))
    cb_calls = []

    tm = TaskManager(tokenizer=t, max_seq_len=16)
    tm.add_task("long", stream_callback=lambda tok: cb_calls.append(tok))
    assert cb_calls[0] is STOP
    assert len(tm.waiting_queue) == 0


def test_task_manager_remove_task():
    tm = TaskManager(tokenizer=_make_mock_tokenizer())
    tid = tm.add_task("test")
    tm.remove_task(tid)
    assert len(tm.waiting_queue) == 0


def test_task_manager_remove_active_task():
    tm = TaskManager(tokenizer=_make_mock_tokenizer())
    tid = tm.add_task("test")
    tasks = tm.pull_candidates(1)
    tm.activate(tasks[0])
    assert len(tm.active_tasks) == 1
    removed = tm.remove_task(tid)
    assert len(removed) == 1
    assert len(tm.active_tasks) == 0


def test_task_manager_pull_candidates_fifo():
    tm = TaskManager(tokenizer=_make_mock_tokenizer())
    tm.add_task("a")
    tm.add_task("b")
    tm.add_task("c")
    pulled = tm.pull_candidates(2)
    assert len(pulled) == 2
    assert pulled[0].prompt_ids == [1, 2, 3, 4, 5]
    assert len(tm.waiting_queue) == 1


def test_task_manager_activate():
    tm = TaskManager(tokenizer=_make_mock_tokenizer())
    tm.add_task("test")
    task = tm.pull_candidates(1)[0]
    tm.activate(task)
    assert task.status == TaskStatus.RUNNING
    assert task in tm.active_tasks


def test_task_manager_return_to_waiting():
    tm = TaskManager(tokenizer=_make_mock_tokenizer())
    tm.add_task("a")
    tm.add_task("b")
    t1 = tm.pull_candidates(1)[0]
    tm.return_to_waiting([t1])
    assert len(tm.waiting_queue) == 2
    assert tm.waiting_queue[0] == t1


def test_task_manager_remove_finished_aborted():
    tm = TaskManager(tokenizer=_make_mock_tokenizer())
    tm.add_task("test")
    task = tm.pull_candidates(1)[0]
    tm.activate(task)
    task.status = TaskStatus.ABORTED
    finished = tm.remove_finished_tasks([0])
    assert len(finished) == 1
    assert len(tm.active_tasks) == 0


def test_task_manager_remove_finished_stop_id():
    tm = TaskManager(tokenizer=_make_mock_tokenizer())
    tm.add_task("test")
    task = tm.pull_candidates(1)[0]
    tm.activate(task)
    task.output_ids = [0]
    task.output_tokens = 1
    finished = tm.remove_finished_tasks([0])
    assert len(finished) == 1
    assert task.status == TaskStatus.FINISHED
    assert len(tm.active_tasks) == 0


def test_task_manager_has_work():
    tm = TaskManager(tokenizer=_make_mock_tokenizer())
    assert not tm.has_work()
    tm.add_task("test")
    assert tm.has_work()


def test_task_manager_wake():
    import threading

    tm = TaskManager(tokenizer=_make_mock_tokenizer())
    called = threading.Event()

    def waiter():
        tm.wait_for_tasks(timeout=5.0)
        called.set()

    t = threading.Thread(target=waiter)
    t.start()
    import time

    time.sleep(0.05)
    tm.wake()
    t.join(timeout=2.0)
    assert called.is_set()


def test_task_manager_get_stats():
    tm = TaskManager(tokenizer=_make_mock_tokenizer())
    tm.add_task("test")
    stats = tm.get_stats()
    assert stats["total_tasks"] == 1
    assert stats["waiting_queue"] == 1
    assert stats["active_tasks"] == 0
