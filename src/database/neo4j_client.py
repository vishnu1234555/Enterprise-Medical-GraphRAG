import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load .env from project root (works when Streamlit cwd differs from repo root)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv()


def get_driver():
    # Default matches Neo4j Browser “local” URI. Also valid: bolt://127.0.0.1:7687
    # Docker Compose: set NEO4J_URI=bolt://neo4j:7687 (or neo4j://neo4j:7687)
    uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "12345678")
    return GraphDatabase.driver(uri, auth=(user, password))


def session_kwargs():
    """Optional Neo4j 4+ database name (e.g. neo4j vs system)."""
    db = os.getenv("NEO4J_DATABASE", "").strip()
    return {"database": db} if db else {}
