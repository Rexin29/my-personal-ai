# Educational Agent (Google ADK + RAG + Syllabus DB)

This project provides a Python ADK-based agent that routes student queries to either:
- a Vector Retrieval (RAG) tool (Chroma / Pinecone / Vertex), or
- a Structured Syllabus Database (Postgres / SQLite).

Files:
- main.py — entrypoint and CLI
- agent.py — RootOrchestratorAgent with routing & composition logic
- tools/vector_tool.py — vector DB wrapper (Chroma / Pinecone / Vertex)
- tools/db_tool.py — syllabus DB wrapper (SQLAlchemy)
- requirements.txt, .env.example, README.md

Quick start
1. Install dependencies
   python -m pip install -r requirements.txt

2. Set environment variables (copy .env.example -> .env) and populate:
   - GOOGLE_API_KEY (required)
   - VECTOR_DB_TYPE and its credentials / endpoint
   - DATABASE_URL (e.g. sqlite:///syllabus.db or postgres://user:pw@host/dbname)

3. Prepare your data:
   - Structured syllabus: create a 'subjects' table with columns like
     id, semester (int), subject_name (text), course_code (text), topics (text/json), credits
     You can use SQL scripts or ORM migrations to populate it.

   - Notes (RAG pipeline): extract text from lectures/slides/PDFs, chunk, create embeddings,
     and upsert into your chosen vector DB. For Chroma, create a collection "semester_notes"
     and set metadata like {"source": filename} for each doc chunk.

4. Run the CLI:
   python main.py

Design notes
- The agent class uses the model to classify routing decisions into SYLLABUS vs NOT_SYLLABUS.
  This helps avoid hallucinating syllabus details for queries that should be answered directly from structured data.

- The RAG prompt includes retrieved passages and asks the model to cite them.

Next steps & production considerations
- Implement authentication, rate-limiting, and usage logging.
- Add tests and CI.
- Move to a server with an API if you want a web interface.
- Implement secure key & secret management (don't store secrets in plaintext).
- Hook up an embeddings pipeline (Gemini embeddings or Vertex embeddings) to compute vectors.
