import logging
import os
from typing import Iterable, List, Tuple

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("vectorize")

# AUDIT PATCH 1: Direct IPv4 Bolt protocol to bypass routing crashes
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
INDEX_NAME = "disease_name_embeddings"
VECTOR_DIMENSIONS = 384

# AUDIT PATCH 2: Reduced database write batch to save RAM
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))

def chunked(items: List[Tuple[str, str]], size: int) -> Iterable[List[Tuple[str, str]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]

def create_vector_index(driver) -> None:
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
    with driver.session() as session:
        session.run(query).consume()
    logger.info("Ensured vector index exists: %s", INDEX_NAME)

def fetch_unembedded_diseases(driver) -> List[Tuple[str, str]]:
    query = """
    MATCH (d:Disease)
    WHERE d.embedding IS NULL
    RETURN d.id AS id, d.name AS name
    """
    with driver.session() as session:
        result = session.run(query)
        rows = [(record["id"], record["name"]) for record in result if record["name"]]
    logger.info("Fetched %d Disease nodes without embeddings", len(rows))
    return rows

def write_embeddings(driver, rows: List[dict]) -> None:
    query = """
    UNWIND $rows AS row
    MATCH (d:Disease {id: row.id})
    SET d.embedding = row.embedding
    """
    with driver.session() as session:
        session.run(query, rows=rows).consume()

def main() -> None:
    logger.info("Loading model: %s on CPU", MODEL_NAME)
    # AUDIT PATCH 3: Explicitly lock to CPU so it doesn't hunt for a non-existent GPU
    model = SentenceTransformer(MODEL_NAME, device='cpu')
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        create_vector_index(driver)
        diseases = fetch_unembedded_diseases(driver)
        if not diseases:
            logger.info("No Disease nodes require embedding. Exiting.")
            return

        total_written = 0
        for batch in chunked(diseases, BATCH_SIZE):
            ids = [item[0] for item in batch]
            names = [item[1] for item in batch]

            embeddings = model.encode(
                names,
                # AUDIT PATCH 4: Micro-batching for i3 Processor survival
                batch_size=16, 
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            payload = [
                {"id": node_id, "embedding": vector.tolist()}
                for node_id, vector in zip(ids, embeddings)
            ]
            write_embeddings(driver, payload)
            total_written += len(payload)
            logger.info("Wrote embeddings for %d/%d nodes", total_written, len(diseases))

        logger.info("Embedding job completed successfully. Total updated: %d", total_written)
    finally:
        driver.close()

if __name__ == "__main__":
    main()