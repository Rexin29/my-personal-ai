"""
Entry point for the Root Orchestrator educational agent.
Run: python main.py
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()  # loads .env if present

from google_adk import Model  # assumed google-adk client
from agent import RootOrchestratorAgent
from tools.vector_tool import VectorRetrievalTool
from tools.db_tool import SyllabusDBTool

# Environment / defaults
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")  # must be set
VECTOR_DB_TYPE = os.environ.get("VECTOR_DB_TYPE", "chroma")  # chroma|pinecone|vertex
VECTOR_DB_ENDPOINT = os.environ.get("VECTOR_DB_ENDPOINT", "")
VECTOR_DB_API_KEY = os.environ.get("VECTOR_DB_API_KEY", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///syllabus.db")

if not GOOGLE_API_KEY:
    print("ERROR: GOOGLE_API_KEY is required in environment.")
    sys.exit(1)

# Initialize model client (google-adk)
model = Model(api_key=GOOGLE_API_KEY, model_name=GEMINI_MODEL)

# Initialize tools
vector_tool = VectorRetrievalTool(
    vector_db_type=VECTOR_DB_TYPE,
    endpoint=VECTOR_DB_ENDPOINT,
    api_key=VECTOR_DB_API_KEY,
    model_client=model,
)

db_tool = SyllabusDBTool(database_url=DATABASE_URL, model_client=model)

# System instruction (robust) used by Root Orchestrator
SYSTEM_INSTRUCTIONS = """
You are the Root Orchestrator agent for a college educational assistant.
You MUST decide whether a user's query is strictly a structured syllabus request
(e.g., "What are the topics in Semester 4 Data Structures?", "Show course code CS402 and credits")
or an unstructured knowledge/notes request that requires RAG (e.g., "Explain Dijkstra's algorithm from my notes", "Summarize lecture 3").
If it's a syllabus query return a routing decision 'SYLLABUS' and the minimal structured parameters needed (semester, subject name or code).
If it's a knowledge query return 'NOT_SYLLABUS'. Use the structured database tool only for syllabus-like queries; use the RAG tool for conceptual explanations, examples, or where the user explicitly references notes or lectures.
When uncertain, prefer RAG but mention confidence and cite retrieved notes or DB rows.
"""

agent = RootOrchestratorAgent(
    model_client=model,
    rag_tool=vector_tool,
    db_tool=db_tool,
    system_instructions=SYSTEM_INSTRUCTIONS,
)

def cli_loop():
    print("Welcome to the Educational Agent CLI. Type 'exit' to quit.")
    while True:
        query = input("\nStudent> ").strip()
        if query.lower() in ("exit", "quit"):
            break
        try:
            response = agent.handle_query(query)
            print("\nAgent>\n")
            print(response)
        except Exception as e:
            print(f"Agent encountered an error: {e}")

if __name__ == "__main__":
    cli_loop()
