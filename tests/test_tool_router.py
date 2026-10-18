from types import SimpleNamespace

from knowledge.processor.query_process.nodes import node_tool_router


class FakeBoundModel:
    def invoke(self, messages):
        return SimpleNamespace(tool_calls=[{"name": "scene_search"}, {"name": "web_search"}])


class FakeModel:
    def bind_tools(self, tools, tool_choice):
        assert tool_choice == "auto"
        return FakeBoundModel()


def test_model_autonomously_selects_tools(monkeypatch):
    monkeypatch.setattr(node_tool_router, "get_llm_client", lambda: FakeModel())
    state = {"original_query": "查询最新发布场景"}
    node_tool_router.NodeToolRouter().process(state)
    assert state["selected_tools"] == ["scene_search", "web_search"]
