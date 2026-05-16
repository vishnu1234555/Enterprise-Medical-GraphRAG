"""Neo4j vector index + embedding read/write (no ML imports)."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable, List, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase

from src.database.neo4j_client import session_kwargs

# Load environment variables
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

log = logging.getLogger(__name__)

# Constants for Optimization
INDEX_NAME = "disease_name_embeddings"
VECTOR_DIMENSIONS = 384
BATCH_SIZE_DEFAULT = 5000  # Optimized for i3/16GB RAM overhead

def create_vector_index(driver) -> None:
    """Ensures the Neo4j Vector Index is present."""
    query = f"""
    CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS
    FOR (d:Disease) ON (d.embedding)
    OPTIONS {{
      indexConfig: {{
        `vector.dimensions`: {VECTOR_DIMENSIONS},
        `vector.similarity_function`: 'cosine'
      }}
    }}
    """
    with driver.session(**session_kwargs()) as session:
        session.run(query).consume()
    log.info("Ensured vector index exists: %s", INDEX_NAME)


def fetch_unembedded_diseases(driver) -> List[Tuple[str, str]]:
    """Retrieves all Disease nodes missing embeddings."""
    query = """
    MATCH (d:Disease)
    WHERE d.embedding IS NULL
    RETURN d.id AS id, d.name AS name
    """
    with driver.session(**session_kwargs()) as session:
        result = session.run(query)
        rows = [(record["id"], record["name"]) for record in result if record["name"]]
    log.info("Fetched %d Disease nodes without embeddings", len(rows))
    return rows


def write_embeddings(driver, rows: List[dict[str, Any]]) -> None:
    """Uses high-speed UNWIND to batch-update node embeddings."""
    query = """
    UNWIND $rows AS row
    MATCH (d:Disease {id: row.id})
    SET d.embedding = row.embedding
    """
    with driver.session(**session_kwargs()) as session:
        session.run(query, rows=rows).consume()


def chunked(items: List[Any], size: int = BATCH_SIZE_DEFAULT) -> Iterable[List[Any]]:
    """Yield successive n-sized chunks from items."""
    for i in range(0, len(items), size):
        yield items[i : i + size]