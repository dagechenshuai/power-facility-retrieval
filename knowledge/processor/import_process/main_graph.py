"""知识库导入工作流 —— 7 节点串行流水线"""

from langgraph.constants import END
from langgraph.graph import StateGraph

from knowledge.processor.import_process.nodes.node_entry import NodeEntry
from knowledge.processor.import_process.nodes.node_pdf_to_md import NodePDFToMD
from knowledge.processor.import_process.nodes.node_md_img import NodeMDImg
from knowledge.processor.import_process.nodes.node_document_split import NodeDocumentSplit
from knowledge.processor.import_process.nodes.node_facility_name import NodeFacilityNameRecognition
from knowledge.processor.import_process.nodes.node_bge_embedding import NodeBGEEmbedding
from knowledge.processor.import_process.nodes.node_import_milvus import NodeImportMilvus
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.persistence import create_checkpointer


class KBImportWorkFlow:
    """知识库导入工作流 —— 搭流水线，7 个节点串起来跑"""

    def __init__(self):
        self._compiled_graph = None

    @property
    def compiled_graph(self):
        """懒加载：第一次调用才编译，之后用缓存"""
        if self._compiled_graph is None:
            self._compiled_graph = self.build_graph()
        return self._compiled_graph

    def build_graph(self):
        """搭流水线：注册节点 + 连接边"""
        graph = StateGraph(ImportGraphState)

        # ① 注册 7 个节点
        graph.add_node("node_entry", NodeEntry())
        graph.add_node("node_pdf_to_md", NodePDFToMD())
        graph.add_node("node_md_img", NodeMDImg())
        graph.add_node("node_document_split", NodeDocumentSplit())
        graph.add_node("node_facility_name", NodeFacilityNameRecognition())
        graph.add_node("node_bge_embedding", NodeBGEEmbedding())
        graph.add_node("node_import_milvus", NodeImportMilvus())

        # ② 入口
        graph.set_entry_point("node_entry")

        # ③ 入口判断：PDF → 走转换链，MD → 直接跳到图片处理
        graph.add_conditional_edges(
            "node_entry",
            self._route_after_entry,
            {
                "node_pdf_to_md": "node_pdf_to_md",
                "node_md_img": "node_md_img",
                END: END,
            }
        )

        # ④ 串行流水线
        graph.add_edge("node_pdf_to_md", "node_md_img")
        graph.add_edge("node_md_img", "node_document_split")
        graph.add_edge("node_document_split", "node_facility_name")
        graph.add_edge("node_facility_name", "node_bge_embedding")
        graph.add_edge("node_bge_embedding", "node_import_milvus")
        graph.add_edge("node_import_milvus", END)

        return graph.compile(checkpointer=create_checkpointer())

    @staticmethod
    def _route_after_entry(state: ImportGraphState) -> str:
        """入口路由：PDF→转换，MD→直接处理"""
        if state.get("is_pdf_read_enabled"):
            return "node_pdf_to_md"
        elif state.get("is_md_read_enabled"):
            return "node_md_img"
        else:
            return END

    def run(self, state: ImportGraphState | None, stream: bool = False, thread_id: str | None = None):
        """启动流水线"""
        resolved_thread_id = thread_id or (state or {}).get("task_id", "import-default")
        config = {"configurable": {"thread_id": resolved_thread_id}}
        if stream:
            return self.compiled_graph.stream(state, config=config, stream_mode="updates")
        else:
            return self.compiled_graph.invoke(state, config=config)
