"""查询失败、并发隔离、上下文预算及检索反馈回归。"""

import asyncio
import threading
import uuid
from types import SimpleNamespace

import httpx
import pytest

from knowledge.processor.query_process.base import NodeBase
from knowledge.processor.query_process.nodes import node_review_retrieval as review
from knowledge.processor.query_process.nodes import node_tool_router as router
from knowledge.processor.query_process.nodes.node_answer_output import NodeAnswerOutput
from knowledge.processor.query_process.nodes.node_search_bm25 import NodeSearchBm25
from knowledge.utils import persistence, sse_utils
from web.api import query_service as api


class FakeWorkflow:
    def run(self, *args, **kwargs):
        raise RuntimeError("模拟模型失败")


def test_query_failure_returns_failed_http_status(monkeypatch):
    monkeypatch.setattr(api, "KBQueryWorkflow", FakeWorkflow)

    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/query", json={"query": "测试", "is_stream": False}
            )
            assert response.status_code == 502
            assert response.json()["status"] == "failed"
            task = persistence.get_task(response.json()["task_id"])
            assert task["status"] == "failed"

    asyncio.run(request())


def test_empty_model_answer_is_not_completed(monkeypatch):
    class EmptyWorkflow:
        def run(self, *args, **kwargs):
            yield {"answer": ""}

    monkeypatch.setattr(api, "KBQueryWorkflow", EmptyWorkflow)
    task_id = str(uuid.uuid4())
    persistence.upsert_task(task_id, "query", "processing")
    api.run_query_graph(task_id, "session", "测试", False)
    assert persistence.get_task(task_id)["status"] == "failed"


def test_nonstream_query_does_not_block_health(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    class SlowWorkflow:
        def run(self, *args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("健康检查未能并发执行")
            yield {"answer": "测试回答", "result_status": "answered"}

    monkeypatch.setattr(api, "KBQueryWorkflow", SlowWorkflow)

    async def request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://test"
        ) as client:
            pending = asyncio.create_task(client.post("/query", json={"query": "测试"}))
            try:
                while not entered.is_set():
                    await asyncio.sleep(0.01)
                assert not pending.done()
                response = await asyncio.wait_for(client.get("/health"), 1)
                assert response.status_code == 200
            finally:
                release.set()
            assert (await pending).status_code == 200

    asyncio.run(request())


def test_parallel_node_returns_only_its_changes():
    class ExampleNode(NodeBase):
        def process(self, state):
            state["bm25_chunks"] = ["new"]
            return state

    original = {"embedding_chunks": [], "bm25_chunks": []}
    assert ExampleNode()(original) == {"bm25_chunks": ["new"]}
    assert original["bm25_chunks"] == []


def test_oversized_first_document_still_provides_evidence():
    text, remaining = NodeAnswerOutput()._format_reranked_docs(
        [{"chunk_id": "p1", "content": "有效资料" * 1000}], 200
    )
    assert "有效资料" in text
    assert "[已截断]" in text
    assert remaining >= 0


def test_model_tool_query_reaches_retriever(monkeypatch):
    model = SimpleNamespace(
        bind_tools=lambda *a, **k: SimpleNamespace(
            invoke=lambda *a: SimpleNamespace(
                tool_calls=[{"name": "bm25_search", "args": {"query": "风机叶片检查"}}]
            )
        )
    )
    monkeypatch.setattr(router, "get_llm_client", lambda: model)
    state = router.NodeToolRouter().process({"original_query": "它多久查一次"})
    received = []
    monkeypatch.setattr(
        "knowledge.processor.query_process.nodes.node_search_bm25.search_bm25",
        lambda query, *a, **k: received.append(query) or [],
    )
    NodeSearchBm25().process(state)
    assert received == ["风机叶片检查"]


@pytest.mark.parametrize(
    "action,round_number,expected",
    [
        ("retry", 1, "retry"),
        ("retry", 2, "stop"),
        ("answer", 1, "answer"),
        ("clarify", 1, "clarify"),
        ("stop", 1, "stop"),
    ],
)
def test_review_actions_and_limit(monkeypatch, action, round_number, expected):
    decision = review.RetrievalDecision(
        action=action, reason="测试依据", query="新查询", question="具体哪台设备？"
    )
    model = SimpleNamespace(
        with_structured_output=lambda *a: SimpleNamespace(invoke=lambda *a: decision)
    )
    monkeypatch.setattr(review, "get_llm_client", lambda: model)
    state = review.NodeReviewRetrieval().process(
        {
            "original_query": "检查周期",
            "retrieval_round": round_number,
            "reranked_docs": [{"chunk_id": "p1", "content": "测试资料"}],
        }
    )
    assert state["next_action"] == expected
    if action == "retry" and round_number == 2:
        assert state["result_status"] == "incomplete"
    assert len(state["decision_trace"]) == 1


class ConnectedRequest:
    async def is_disconnected(self):
        return False


def test_two_sse_subscribers_receive_all_events():
    async def scenario():
        task_id = str(uuid.uuid4())
        first = sse_utils.sse_generator(task_id, ConnectedRequest())
        second = sse_utils.sse_generator(task_id, ConnectedRequest())

        async def collect(stream):
            return [event async for event in stream]

        tasks = [
            asyncio.create_task(collect(first)),
            asyncio.create_task(collect(second)),
        ]
        await asyncio.sleep(0.02)
        sse_utils.push_to_session(task_id, "delta", {"delta": "A"})
        sse_utils.push_to_session(task_id, "final", {"answer": "A"})
        results = await asyncio.wait_for(asyncio.gather(*tasks), 2)
        assert results[0] == results[1]
        assert len(results[0]) == 2

    asyncio.run(scenario())


def test_sse_replay_skips_old_failure_after_resume():
    task_id = str(uuid.uuid4())
    sse_utils.push_to_session(task_id, "error", {"error": "旧失败"})
    sse_utils.push_to_session(task_id, "ready", {"reset": True})
    sse_utils.push_to_session(task_id, "final", {"answer": "恢复成功"})

    async def collect():
        return [
            event
            async for event in sse_utils.sse_generator(task_id, ConnectedRequest())
        ]

    events = asyncio.run(collect())
    assert len(events) == 2
    assert "恢复成功" in events[-1]
    assert not any("旧失败" in event for event in events)


@pytest.mark.parametrize("has_evidence", [True, False])
def test_graph_observes_empty_result_then_retries(monkeypatch, has_evidence):
    from knowledge.processor.query_process.main_graph import KBQueryWorkflow

    decisions = []
    model = SimpleNamespace(
        bind_tools=lambda *a, **k: SimpleNamespace(
            invoke=lambda *a: SimpleNamespace(
                tool_calls=[
                    {"name": "vector_search", "args": {"query": "叶片"}},
                    {"name": "bm25_search", "args": {"query": "叶片检查"}},
                ]
            )
        )
    )
    monkeypatch.setattr(router, "get_llm_client", lambda: model)

    def decide(messages):
        import json

        observation = json.loads(messages[-1][1])
        decisions.append(observation)
        return review.RetrievalDecision(
            action="answer" if observation["evidence"] else "retry",
            reason="已有证据" if observation["evidence"] else "需要换词补查",
            query="叶片检查周期",
        )

    monkeypatch.setattr(
        review,
        "get_llm_client",
        lambda: SimpleNamespace(
            with_structured_output=lambda *a: SimpleNamespace(invoke=decide)
        ),
    )
    workflow = KBQueryWorkflow()
    monkeypatch.setattr(
        workflow.node_facility_confirm,
        "process",
        lambda s: {**s, "rewritten_query": s["original_query"]},
    )

    def vector(state):
        state["embedding_chunks"] = (
            []
            if state["retrieval_round"] == 1 or not has_evidence
            else [{"chunk_id": "parent:p1", "content": "合成资料：检查周期见维护计划"}]
        )
        return state

    def bm25(state):
        state["bm25_chunks"] = (
            []
            if state["retrieval_round"] == 1 or not has_evidence
            else [{"chunk_id": "parent:p2", "content": "合成资料：需要现场复核"}]
        )
        return state

    monkeypatch.setattr(workflow.node_search_embedding, "process", vector)
    monkeypatch.setattr(workflow.node_bm25, "process", bm25)
    monkeypatch.setattr(
        workflow.node_rerank,
        "process",
        lambda s: {**s, "reranked_docs": s["rrf_chunks"]},
    )
    monkeypatch.setattr(
        workflow.node_answer, "process", lambda s: {**s, "answer": "合成测试回答"}
    )
    state = workflow.run(
        {"original_query": "检查周期", "answer": ""}, thread_id=str(uuid.uuid4())
    )
    assert state["retrieval_round"] == 2
    assert len(decisions) == 2
    assert not decisions[0]["evidence"]
    if has_evidence:
        assert {doc["id"] for doc in decisions[1]["evidence"]} == {
            "parent:p1",
            "parent:p2",
        }
        assert state["result_status"] == "answered"
    else:
        assert state["result_status"] == "incomplete"
        assert state["next_action"] == "stop"


def test_vector_and_bm25_share_parent_identity():
    from knowledge.processor.query_process.nodes.node_rrf import NodeRrf
    from knowledge.processor.query_process.nodes.node_search_embedding import (
        NodeSearchEmbedding,
    )
    from knowledge.utils.bm25_store import replace_document, search_bm25

    parent = {
        "parent_id": "same-parent",
        "content": "风机维护",
        "parent_content": "风机维护测试资料",
    }
    replace_document("identity-test", [parent])
    vector = NodeSearchEmbedding._parents([{"entity": parent}])
    bm25 = search_bm25("风机维护")
    matching = [doc for doc in bm25 if doc["parent_id"] == "same-parent"]
    assert matching[0]["chunk_id"] == vector[0]["chunk_id"]
    assert len(NodeRrf()._rrf_merge([(vector, 1.0), (matching, 1.0)])) == 1


def test_sse_disconnect_does_not_interrupt_other_subscriber():
    async def scenario():
        task_id = str(uuid.uuid4())
        sse_utils.push_to_session(task_id, "delta", {"delta": "A"})
        first = sse_utils.sse_generator(task_id, ConnectedRequest())
        second = sse_utils.sse_generator(task_id, ConnectedRequest())
        assert await anext(first) == await anext(second)
        await first.aclose()
        sse_utils.push_to_session(task_id, "final", {"answer": "AB"})
        assert '"AB"' in await asyncio.wait_for(anext(second), 1)
        await second.aclose()

    asyncio.run(scenario())


def test_sse_cursor_replays_only_unseen_events():
    task_id = str(uuid.uuid4())
    cursor = persistence.append_event(task_id, "delta", {"delta": "A"})
    persistence.append_event(task_id, "final", {"answer": "AB"})

    async def collect():
        return [
            item
            async for item in sse_utils.sse_generator(
                task_id, ConnectedRequest(), cursor
            )
        ]

    events = asyncio.run(collect())
    assert len(events) == 1 and "event: final" in events[0]


def test_missing_evidence_cannot_be_marked_answered(monkeypatch):
    decision = review.RetrievalDecision(action="answer", reason="模型误判")
    monkeypatch.setattr(
        review,
        "get_llm_client",
        lambda: SimpleNamespace(
            with_structured_output=lambda *a: SimpleNamespace(
                invoke=lambda *a: decision
            )
        ),
    )
    state = review.NodeReviewRetrieval().process(
        {"original_query": "测试", "reranked_docs": []}
    )
    assert state["result_status"] == "insufficient_evidence"
    assert state["next_action"] == "stop"


def test_resume_keeps_completed_checkpoint_answer(monkeypatch):
    class CompletedWorkflow:
        def compile(self):
            return SimpleNamespace(
                get_state=lambda *a: SimpleNamespace(
                    values={
                        "answer": "检查点中的已完成回答",
                        "result_status": "answered",
                    }
                )
            )

        def run(self, *a, **k):
            return iter(())

    monkeypatch.setattr(api, "KBQueryWorkflow", CompletedWorkflow)
    task_id = str(uuid.uuid4())
    persistence.upsert_task(task_id, "query", "interrupted")
    api.run_query_graph(task_id, "session", "测试", False, True)
    task = persistence.get_task(task_id)
    assert task["status"] == "completed"
    assert task["result"]["values"]["answer"] == "检查点中的已完成回答"
