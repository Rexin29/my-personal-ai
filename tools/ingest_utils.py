"""
Helper utilities for ingestion: PDF extraction, chunking, embedding generation, and Chroma upsert.
"""
import os
from typing import List, Optional

# PDF text extraction
try:
    from pdfminer.high_level import extract_text
except Exception:
    extract_text = None


def extract_text_from_pdf(pdf_path: str) -> str:
    if extract_text is None:
        raise RuntimeError("pdfminer.six is required for PDF extraction. Install pdfminer.six in requirements.")
    text = extract_text(pdf_path)
    return text or ""


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Naive character-based chunking with overlap.
    """
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, end)
    return chunks


def _embed_with_gemini(texts: List[str], model_client) -> List[List[float]]:
    # Assumes model_client has an embeddings method accepting a list of strings
    if model_client is None:
        raise RuntimeError("Model client for Gemini embeddings not provided.")
    if hasattr(model_client, "embeddings"):
        # Some ADK clients accept a list or single string; adapt if necessary
        try:
            emb_resp = model_client.embeddings(texts)
            return emb_resp
        except Exception:
            # try per-item
            return [model_client.embeddings(t) for t in texts]
    raise RuntimeError("Provided model client does not implement embeddings()")


def _embed_with_sentence_transformers(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence-transformers not installed: " + str(e))
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True).tolist()


def get_embeddings(texts: List[str], model_client: Optional[object] = None, batch_size: int = 32) -> List[List[float]]:
    """
    Generate embeddings for a list of texts. Preferred: Gemini embeddings via model_client. Fallback: sentence-transformers.
    """
    if model_client is not None:
        try:
            return _embed_with_gemini(texts, model_client)
        except Exception as e:
            print("Gemini embeddings failed, falling back to sentence-transformers:", e)
    # Fallback
    return _embed_with_sentence_transformers(texts)


def upsert_to_chroma(ids: List[str], documents: List[str], metadatas: List[dict], embeddings: List[List[float]]):
    try:
        import chromadb
        from chromadb.config import Settings
    except Exception as e:
        raise RuntimeError("chromadb is required for upsert: " + str(e))

    endpoint = os.environ.get("VECTOR_DB_ENDPOINT") or "http://localhost"
    try:
        client = chromadb.Client(Settings(chroma_api_impl="chromadb.http", chroma_server_host=endpoint, chroma_server_http_port=8000))
        collection = client.get_or_create_collection("semester_notes")
    except Exception as e:
        # Try local client without HTTP mode
        client = chromadb.Client()
        collection = client.get_or_create_collection("semester_notes")

    # Chroma's add/upsert API varies by version. We'll try upsert, else add.
    try:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    except Exception:
        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    print(f"Upserted {len(ids)} chunks into Chroma collection 'semester_notes'.")
