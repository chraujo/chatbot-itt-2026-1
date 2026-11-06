from typing import TypedDict, Optional, List
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage

from .chains import get_triage_chain, get_rag_chain
from .vectorDB import VectorDB
from ..config import Settings

class AgentState(TypedDict, total=False):
    question: str
    history: List[dict]
    triage: dict
    answer: Optional[str]
    citations: List[dict]
    rag_success: bool
    final_action: str

class Citation(TypedDict):
    source: str
    page: Optional[int]
    sheet: Optional[str]
    snippet: str

class GraphResponse(TypedDict):
    response: str
    source_documents: List[Citation]

class ITTGraph:
    def __init__(self, settings: Settings, vector_db: VectorDB):
        self.settings = settings
        self.vector_db = vector_db
        self.triage_chain = get_triage_chain(settings)
        self.rag_chain = get_rag_chain(settings)
        self.graph = self._build_graph()
    
    def _node_triage(self, state: AgentState) -> AgentState:
        question = state.get("question", "")
        messages = [
            (HumanMessage if message['role'] == 'user' else AIMessage)(content=message['content'])
            for message in state.get('history', [])
        ]
        triage_result = self.triage_chain.invoke(messages + [HumanMessage(content=question)])
        return {"triage": triage_result.dict()}

    def _node_auto_resolve(self, state: AgentState) -> AgentState:
        triage = state.get("triage", {})
        question = triage.get('pergunta_contextualizada') or state.get("question", "")
        search_query = triage.get('consulta_busca')
        related_docs = self.vector_db.query(search_query or question)

        if not related_docs:
            return {"answer": "Não sei.", "citations": [], "rag_success": False}

        for doc in related_docs:
            doc.metadata['source'] = Path(doc.metadata.get('source', 'Desconhecido')).name
        llm_response = self.rag_chain.invoke({"input": question, "context": related_docs})
        text = (llm_response or "").strip()

        if text.rstrip(".!?") == "Não sei": 
            return {"answer": "Não sei.", "citations": [], "rag_success": False}

        seen = set()
        citations = []
        for doc in related_docs:
            raw_source = doc.metadata.get("source", "Desconhecido")
            source_name = Path(raw_source).name
            raw_page = doc.metadata.get("page")
            page = (raw_page + 1) if isinstance(raw_page, int) else None
            sheet = doc.metadata.get('sheet')
            key = (source_name, page, sheet)
            if key in seen:
                continue
            seen.add(key)
            citations.append({
                "source": source_name,
                "page": page,
                "sheet": sheet,
                "snippet": doc.page_content[:300].strip(),
            })

        return {"answer": text, "citations": citations, "rag_success": True}

    def _node_request_info(self, state: AgentState) -> AgentState:
        missing_fields = state.get("triage", {}).get("campos_faltantes", [])
        details = ", ".join(missing_fields) if missing_fields else "mais detalhes sobre sua dúvida"
        return {
            "answer": f"Para te ajudar melhor, por favor, forneça {details}.",
            "citations": [],
            "final_action": "REQUEST_INFO"
        }

    def _decide_after_triage(self, state: AgentState) -> str:
        decision = state.get("triage", {}).get("decisao", "").lower()
        if decision == "auto_resolver":
            return "auto_resolve"
        return "request_info"

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("triage_node", self._node_triage)
        workflow.add_node("auto_resolve", self._node_auto_resolve)
        workflow.add_node("request_info", self._node_request_info)

        workflow.add_edge(START, "triage_node")
        workflow.add_conditional_edges("triage_node", self._decide_after_triage, {
            "auto_resolve": "auto_resolve",
            "request_info": "request_info"
        })
        workflow.add_edge("auto_resolve", END)
        workflow.add_edge("request_info", END)

        return workflow.compile()
    
    def invoke(self, question: str, history: Optional[List[dict]] = None) -> GraphResponse:
        result = self.graph.invoke({"question": question, "history": history or []})
        return {
            "response": result.get("answer", ""),
            "source_documents": result.get("citations", []),
        }
