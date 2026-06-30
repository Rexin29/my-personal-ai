"""
Vector retrieval tool wrapper supporting Chroma, Pinecone, Vertex (selectable)
Provides .search(query, top_k) -> List[{"source":..., "text":...}]
"""
import os
from typing import List, Dict, Any

class VectorRetrievalTool:
    def __init__(self, vector_db_type="chroma", endpoint=None, api_key=None, model_client=None):
        self.type = (vector_db_type or "chroma").lower()
        self.endpoint = endpoint
        self.api_key = api_key
        self.model_client = model_client  # may be used to calculate embeddings if needed

        # Lazy import / validate client libraries
        if self.type == "chroma":
            try:
                import chromadb
                from chromadb.config import Settings
                self.chroma = chromadb.Client(Settings(chroma_api_impl="chromadb.http", chroma_server_host=endpoint or "http://localhost", chroma_server_http_port=8000))
                self.collection = None  # set during ingestion or externally
            except Exception as e:
                raise RuntimeError("Chroma client library missing or failed to init: " + str(e))
        elif self.type == "pinecone":
            try:
                import pinecone
                pinecone.init(api_key=api_key, environment=endpoint)
                self.pinecone = pinecone
                self.index = None
            except Exception as e:
                raise RuntimeError("Pinecone client init failed: " + str(e))
        elif self.type == "vertex":
            # Vertex AI Search integration is complex; we expect user to provide an HTTP endpoint or use google cloud client.
            try:
                from google.cloud import aiplatform
                self.aiplatform = aiplatform
            except Exception as e:
                raise RuntimeError("Vertex AI client not available: " + str(e))
        else:
            raise ValueError("Unsupported VECTOR_DB_TYPE: choose chroma|pinecone|vertex")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts: { "source": "filename or id", "text": "excerpt or chunk text" }
        """
        if self.type == "chroma":
            # user must set self.collection externally (or provide default collection name)
            collection = self._ensure_chroma_collection()
            if collection is None:
                return []
            results = collection.query(query_texts=[query], n_results=top_k, include=["metadatas","documents"])
            docs = []
            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                docs.append({"source": meta.get("source", f"doc-{i}"), "text": doc})
            return docs
        elif self.type == "pinecone":
            # index must be set as self.index
            if not getattr(self, "index", None):
                return []
            xq = self._embed(query)
            query_response = self.pinecone.Index(self.index).query(vector=xq, top_k=top_k, include_metadata=True, include_values=False)
            docs = []
            for match in query_response["matches"]:
                docs.append({"source": match["metadata"].get("source","unknown"), "text": match["metadata"].get("text","")})
            return docs
        elif self.type == "vertex":
            # Simple placeholder using Vertex matching
            # Real implementation should call the Vertex Search API / Matching Engine
            raise NotImplementedError("Vertex search helper needs implementation for your project.")
        return []

    def _ensure_chroma_collection(self):
        # If chroma collection already set, return it; otherwise try to load default "semester_notes"
        if getattr(self, "collection", None):
            return self.collection
        try:
            # may raise if collection doesn't exist; user should create ingestion pipeline separately
            self.collection = self.chroma.get_or_create_collection("semester_notes")
            return self.collection
        except Exception:
            return None

    def _embed(self, text: str):
        """
        Generate embedding for the query using available model_client or fallback.
        """
        if self.model_client and hasattr(self.model_client, "embeddings"):
            return self.model_client.embeddings(text)
        # else, raise — user must provide embeddings pipeline
        raise RuntimeError("No embedding generator available. Provide model_client.embeddings or precomputed vectors.")
