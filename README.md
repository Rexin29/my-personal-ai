### Ingestion & Migration additions

I added SQL migration and seed data, plus a notes ingestion script and helper utilities.

New files added:
- db/migrations/001_create_subjects.sql — creates the `subjects` table (SQLite compatible)
- db/seeds/subjects_seed.csv — 5 example rows (id,semester,subject_name,course_code,topics,credits)
- scripts/ingest_notes.py — PDF -> chunk -> embed -> upsert script (Chroma + Gemini embeddings)
- tools/ingest_utils.py — helper functions for PDF extraction, chunking, embeddings, and Chroma upsert

Requirements updated to include pdfminer.six and sentence-transformers.

How to run migration (SQLite):

1. Create the SQLite DB file (if not present) and apply the SQL migration:
   sqlite3 syllabus.db < db/migrations/001_create_subjects.sql

2. Import seed CSV (example using sqlite3):
   sqlite3 syllabus.db
   .mode csv
   .import db/seeds/subjects_seed.csv subjects

3. Verify:
   SELECT count(*) FROM subjects;

How to ingest notes into Chroma with Gemini embeddings

1. Ensure Chroma is running (or accessible). By default the script attempts to reach VECTOR_DB_ENDPOINT or http://localhost:8000.

2. Populate .env with GOOGLE_API_KEY and optionally GEMINI_MODEL.

3. Run the ingestion script on a directory of PDFs:
   python scripts/ingest_notes.py --pdf-dir ./notes --source semester1

Notes & caveats
- The ingestion script prefers Gemini embeddings via google-adk. If the google-adk client or Gemini embeddings are not available, the script falls back to sentence-transformers locally.
- Adjust chunk_size and overlap flags for your document type.
