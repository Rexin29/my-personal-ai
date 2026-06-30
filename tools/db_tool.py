"""
Syllabus DB tool using SQLAlchemy.
Provides higher-level helpers used by the orchestrator.
"""
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, MetaData, Table, select, or_, and_
from sqlalchemy.engine import Engine
import re
import json

class SyllabusDBTool:
    def __init__(self, database_url: str, model_client=None):
        self.database_url = database_url
        self.engine: Engine = create_engine(database_url, echo=False, future=True)
        self.metadata = MetaData(bind=self.engine)
        self.model = model_client
        # Expect a table named 'subjects' with columns: id, semester, subject_name, course_code, topics (text/json), credits
        self._reflect()

    def _reflect(self):
        try:
            self.subjects_table = Table("subjects", self.metadata, autoload_with=self.engine)
        except Exception:
            # Table not found — create a minimal schema suggestion (not auto-creating in production)
            self.subjects_table = None

    def query_by_params(self, params: Dict[str, Any], raw_query: str = "") -> List[Dict[str, Any]]:
        """
        Interpret params to run appropriate structured query.
        If params is empty, try a keyword search fallback.
        """
        if self.subjects_table is None:
            return []

        semester = params.get("semester")
        subject = params.get("subject") or params.get("subject_name") or params.get("course")
        course_code = params.get("course_code") or params.get("code")

        stmt = select(self.subjects_table)
        conds = []
        if semester:
            try:
                sem_num = int(re.sub(r"\D", "", str(semester)))
                conds.append(self.subjects_table.c.semester == sem_num)
            except Exception:
                pass
        if course_code:
            conds.append(self.subjects_table.c.course_code.ilike(f"%{course_code}%"))
        if subject:
            conds.append(self.subjects_table.c.subject_name.ilike(f"%{subject}%"))

        if conds:
            stmt = stmt.where(and_(*conds))
        else:
            # fallback: try natural language matching on subject_name or topics
            keywords = self._extract_keywords(raw_query)
            if keywords:
                kw_conds = [self.subjects_table.c.subject_name.ilike(f"%{k}%") for k in keywords]
                kw_conds += [self.subjects_table.c.topics.ilike(f"%{k}%") for k in keywords if hasattr(self.subjects_table.c, "topics")]
                stmt = stmt.where(or_(*kw_conds))
        with self.engine.connect() as conn:
            results = conn.execute(stmt).mappings().all()
        return [dict(r) for r in results]

    def search_syllabus(self, text_query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Simple text search over subject_name and topics (if available).
        """
        if self.subjects_table is None:
            return []
        keywords = self._extract_keywords(text_query)
        if not keywords:
            return []
        stmt = select(self.subjects_table)
        kw_conds = [self.subjects_table.c.subject_name.ilike(f"%{k}%") for k in keywords]
        if hasattr(self.subjects_table.c, "topics"):
            kw_conds += [self.subjects_table.c.topics.ilike(f"%{k}%") for k in keywords]
        stmt = stmt.where(or_(*kw_conds)).limit(limit)
        with self.engine.connect() as conn:
            results = conn.execute(stmt).mappings().all()
        return [dict(r) for r in results]

    def _extract_keywords(self, text: str) -> List[str]:
        text = (text or "").lower()
        # Simple tokenizer: words of length>=3
        return [w for w in re.findall(r"[a-zA-Z0-9]+", text) if len(w) >= 3][:8]

    # Helper to let pipelines run raw SQL if needed
    def raw_query(self, sql: str) -> List[Dict[str, Any]]:
        with self.engine.connect() as conn:
            results = conn.execute(sql).mappings().all()
            return [dict(r) for r in results]
