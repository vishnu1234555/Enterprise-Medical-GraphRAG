"""
Import embedding vectors from Colab-produced Parquet into local Neo4j.

Expects columns: id (str), embedding (list of 384 floats).

    python -m src.ingestion.import_embeddings_from_parquet
    python -m src.ingestion.import_embeddings_from_parquet --path path/to/disease_embeddings.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.database.neo4j_client import get_driver
from src.ingestion.neo4j_embedding_io import create_vector_index, write_embeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("import_embeddings")

# OPTIMIZED: Increased batch size from 500 to 5000 for high-speed local ingestion
_BATCH = 1000


def _row_to_payload(row) -> dict:
    eid = str(row["id"])
    emb = row["embedding"]
    if hasattr(emb, "tolist"):
        emb = emb.tolist()
    if not isinstance(emb, (list, tuple)) or len(emb) != 384:
        raise ValueError(f"Bad embedding for id={eid!r}: len={getattr(emb, '__len__', lambda: 0)()}")
    return {"id": eid, "embedding": list(map(float, emb))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path",
        type=Path,
        default=ROOT / "data" / "processed" / "disease_embeddings.parquet",
        help="Parquet from Colab (id, embedding)",
    )
    ap.add_argument("--batch", type=int, default=_BATCH, help="Rows per UNWIND write")
    args = ap.parse_args()

    if not args.path.is_file():
        logger.error("File not found: %s", args.path)
        sys.exit(1)

    df = pd.read_parquet(args.path)
    if "id" not in df.columns or "embedding" not in df.columns:
        logger.error("Parquet must have columns: id, embedding. Got: %s", list(df.columns))
        sys.exit(1)

    driver = get_driver()
    try:
        create_vector_index(driver)
        total = len(df)
        written = 0
        for i in range(0, total, args.batch):
            chunk = df.iloc[i : i + args.batch]
            rows = [_row_to_payload(chunk.iloc[j]) for j in range(len(chunk))]
            write_embeddings(driver, rows)
            written += len(rows)
            logger.info("Wrote %d / %d", written, total)
        logger.info("Import complete: %d nodes updated.", written)
    finally:
        driver.close()


if __name__ == "__main__":
    main()