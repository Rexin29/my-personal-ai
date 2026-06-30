"""
Root Orchestrator Agent that routes queries to RAG or structured DB tool.
"""
import json
from typing import List, Dict, Any

class RootOrchestratorAgent:
    def __init__(self, model_client, rag_tool, db_tool, system_instructions: str):
        """
        model_client: an object with .generate(text, **kwargs) or similar method per google-adk
        rag_tool: VectorRetrievalTool instance
        db_tool: SyllabusDBTool instance
        system_instructions: instruction string for classification and answer synthesis
        """
        self.model = model_client
        self.rag = rag_tool
        self.db = db_tool
        self.system_instructions = system_instructions

    def _classify_query(self, user_query: str) -> Dict[str, Any]:
        """
        Use the model to classify the query as SYLLABUS or NOT_SYLLABUS.
        Returns a dict like {"route":"SYLLABUS", "params": {...}, "confidence": 0.95}
        """
        prompt = f"""{self.system_instructions}

User query: """{user_query}"""

Task: Reply with a JSON object only. Fields:
- route: either "SYLLABUS" or "NOT_SYLLABUS"
- confidence: number from 0.0 to 1.0
- params: object with any structured fields you extracted (semester, subject, course_code). If none, use {}.

Respond strictly with JSON - nothing else.
"""
        # Call model; adapt to google-adk Model.generate API
        resp = self.model.generate(prompt, max_output_tokens=200)
        text = resp.text.strip()
        # Try to parse JSON robustly
        try:
            parsed = json.loads(text)
            return parsed
        except Exception:
            # fallback heuristic if model failed to return JSON
            keywords = ["syllabus", "topics", "course code", "semester", "credits", "unit"]
            if any(k in user_query.lower() for k in keywords):
                return {"route": "SYLLABUS", "confidence": 0.6, "params": {}}
            return {"route": "NOT_SYLLABUS", "confidence": 0.6, "params": {}}

    def _synthesize_from_notes(self, user_query: str, docs: List[Dict[str,str]]) -> str:
        """
        Build prompt for RAG completion: include top-k retrieved passages and ask the model to answer,
        citing passage ids/filenames when possible.
        """
        retrieved_texts = "\n\n---\n\n".join(
            f"[{i+1}] source: {d.get('source','unknown')}\n{d.get('text','')}" for i, d in enumerate(docs)
        )
        prompt = f"""You are an educational assistant answering a student's question based on provided lecture notes and semester materials.
If you use exact phrases from the notes, mark them as quotes and include a short citation like [1].
Notes:
{retrieved_texts}

Student question: {user_query}

Answer concisely, use step-by-step explanation when helpful, and include short citations in brackets referring to the numbered notes above.
"""
        resp = self.model.generate(prompt, max_output_tokens=800)
        return resp.text.strip()

    def _format_db_response(self, rows: List[Dict[str,Any]]) -> str:
        if not rows:
            return "I couldn't find matching syllabus entries. Try refining your query (semester or subject name / code)."
        # Pretty print rows
        lines = []
        for r in rows:
            # r expected to be dict-like
            lines.append("• " + "; ".join(f"{k}: {v}" for k, v in r.items()))
        return "\n".join(lines)

    def handle_query(self, user_query: str) -> str:
        """
        Top-level handler: classify, route, and compose a final answer.
        """
        classification = self._classify_query(user_query)
        route = classification.get("route", "NOT_SYLLABUS")
        if route == "SYLLABUS":
            params = classification.get("params", {})
            # Pass params to DB tool. DB tool will attempt to interpret them.
            rows = self.db.query_by_params(params, raw_query=user_query)
            formatted = self._format_db_response(rows)
            return f"(Routed to syllabus DB) Confidence={classification.get('confidence')}\n\n{formatted}"
        else:
            # RAG flow
            docs = self.rag.search(user_query, top_k=5)
            if not docs:
                # fallback: try DB search for any structured matches
                rows = self.db.search_syllabus(user_query)
                if rows:
                    return "(No notes matched; falling back to syllabus DB)\n\n" + self._format_db_response(rows)
                return "I could not find relevant notes or syllabus entries for your query."
            answer = self._synthesize_from_notes(user_query, docs)
            return f"(Routed to RAG over notes) Confidence={classification.get('confidence')}\n\n{answer}"
