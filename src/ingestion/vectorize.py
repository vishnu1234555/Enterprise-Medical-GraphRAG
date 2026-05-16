import logging
import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.database.neo4j_client import get_driver
from src.ingestion.neo4j_embedding_io import (
    chunked,
    create_vector_index,
    fetch_unembedded_diseases,
    write_embeddings,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("vectorize")

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))


def main() -> None:
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    logger.info("Loading model: %s on %s", MODEL_NAME, device)
    model = SentenceTransformer(MODEL_NAME, device=device)
    driver = get_driver()

    try:
        create_vector_index(driver)
        diseases: List[Tuple[str, str]] = fetch_unembedded_diseases(driver)
        if not diseases:
            logger.info("No Disease nodes require embedding. Exiting.")
            return

        total_written = 0
        for batch in chunked(diseases, BATCH_SIZE):
            ids = [item[0] for item in batch]
            names = [item[1] for item in batch]

            encode_bs = min(64, max(8, BATCH_SIZE))
            embeddings = model.encode(
                names,
                batch_size=encode_bs,
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
