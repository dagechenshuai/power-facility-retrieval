import asyncio
import threading
import uuid

from langgraph.graph import END, StateGraph

from knowledge.utils import persistence, sse_utils, task_utils


class ConnectedRequest:
    async def is_disconnected(self):
        return False


def test_task_progress_and_result_are_persisted(tmp_path, monkeypatch):
    task_id = "pytest-task-progress"
    persistence.upsert_task(task_id, "import", "processing", {"path": "demo.md"})
    task_utils._tasks.pop(task_id, None)
    task_utils.add_running_task(task_id, "split")
    task_utils.add_done_task(task_id, "split")
    task_utils.set_task_result(task_id, "answer", "ok")
    task_utils._tasks.pop(task_id, None)

    assert task_utils.get_done_task_list(task_id) == ["split"]
    assert task_utils.get_task_result(task_id, "answer") == "ok"


def test_sse_replays_persisted_token_events():
    task_id = "pytest-sse-replay"
    persistence.append_event(task_id, sse_utils.SSEEvent.DELTA, {"delta": "逐"})
    persistence.append_event(task_id, sse_utils.SSEEvent.DELTA, {"delta": "词"})
    persistence.append_event(task_id, sse_utils.SSEEvent.FINAL, {"answer": "逐词"})

    async def collect():
        return [event async for event in sse_utils.sse_generator(task_id, ConnectedRequest())]

    events = asyncio.run(collect())
    assert sum("event: delta" in event for event in events) == 2
    assert "逐词" in events[-1]


def test_langgraph_checkpoint_can_resume_after_interrupt():
    graph = StateGraph(dict)
    graph.add_node("increment", lambda state: {"value": state["value"] + 1})
    graph.set_entry_point("increment")
    graph.add_edge("increment", END)
    app = graph.compile(
        checkpointer=persistence.CHECKPOINTER,
        interrupt_after=["increment"],
    )
    config = {"configurable": {"thread_id": "pytest-checkpoint-resume"}}

    app.invoke({"value": 0}, config=config)
    assert app.get_state(config).values["value"] == 1
    app.invoke(None, config=config)
    assert app.get_state(config).next == ()


def test_background_thread_can_push_live_sse_tokens():
    async def collect():
        task_id = f"pytest-live-{uuid.uuid4()}"
        sse_utils.create_sse_queue(task_id)

        def produce():
            sse_utils.push_to_session(task_id, sse_utils.SSEEvent.DELTA, {"delta": "A"})
            sse_utils.push_to_session(task_id, sse_utils.SSEEvent.FINAL, {"answer": "A"})

        thread = threading.Thread(target=produce)
        thread.start()
        events = [event async for event in sse_utils.sse_generator(task_id, ConnectedRequest())]
        thread.join()
        return events

    events = asyncio.run(collect())
    assert any("event: delta" in event and '"A"' in event for event in events)
    assert "event: final" in events[-1]
