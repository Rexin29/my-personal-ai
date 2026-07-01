"""
PDF notes ingestion script: extract text, chunk, embed (Gemini embeddings), and upsert to Chroma collection 'semester_notes'.

Usage:
  python scripts/ingest_notes.py --pdf-dir ./notes --source semester1

Environment variables required:
  GOOGLE_API_KEY - used by Gemini embeddings via google-adk
  GEMINI_MODEL - model name (default gemini-2.5-flash)
  VECTOR_DB_ENDPOINT - optional chroma server host (e.g., http://localhost)

This script assumes you have a running Chroma server or local Chroma client available.
"""
import os
import argparse
from pathlib import Path
from typing import List

from tools.ingest_utils import extract_text_from_pdf, chunk_text, get_embeddings, upsert_to_chroma


def discover_pdfs(pdf_dir: Path) -> List[Path]:
    return sorted([p for p in pdf_dir.glob("**/*.pdf")])


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF notes into Chroma using Gemini embeddings")
    parser.add_argument("--pdf-dir", required=True, help="Directory containing PDF files to ingest")
    parser.add_argument("--source", required=True, help="Source label to store in metadata (e.g., semester1)")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Approximate characters per chunk")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap characters between chunks")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embeddings")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    files = discover_pdfs(pdf_dir)
    if not files:
        print(f"No PDF files found in {pdf_dir}")
        return

    # Collect chunks
    documents = []
    metadatas = []
    ids = []
    for pdf_path in files:
        print(f"Processing {pdf_path}")
        text = extract_text_from_pdf(str(pdf_path))
        chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
        for i, chunk in enumerate(chunks):
            doc_id = f"{args.source}:{pdf_path.name}:{i}"
            documents.append(chunk)
            metadatas.append({"source": pdf_path.name, "source_label": args.source, "chunk_index": i})
            ids.append(doc_id)

    print(f"Total chunks: {len(documents)}")

    # Generate embeddings in batches and upsert to Chroma
    model_client = None
    try:
        # Lazy import google-adk Model if available
        from google_adk import Model
        GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
        GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        model_client = Model(api_key=GOOGLE_API_KEY, model_name=GEMINI_MODEL)
    except Exception:
        model_client = None

    embeddings = get_embeddings(documents, model_client=model_client, batch_size=args.batch_size)

    upsert_to_chroma(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
