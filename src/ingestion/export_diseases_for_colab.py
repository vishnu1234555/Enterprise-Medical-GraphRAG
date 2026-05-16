"""
Export Disease id + name rows (embedding IS NULL) to Parquet for Google Colab GPU encoding.

Run on your PC (after load_diseases.py):
    python -m src.ingestion.export_diseases_for_colab

Upload data/processed/diseases_to_embed.parquet to Colab, run notebooks/colab_embed_diseases.ipynb,
download disease_embeddings.parquet, then:
    python -m src.ingestion.import_embeddings_from_parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.database.neo4j_client import get_driver
from src.ingestion.neo4j_embedding_io import fetch_unembedded_diseases


def main() -> None:
    out = ROOT / "data" / "processed" / "diseases_to_embed.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)

    driver = get_driver()
    try:
        rows = fetch_unembedded_diseases(driver)
        if not rows:
            print("[INFO] No diseases with NULL embedding — nothing to export.")
            return
        df = pd.DataFrame(rows, columns=["id", "name"])
        df.to_parquet(out, index=False)
        print(f"[OK] Exported {len(df)} rows to {out}")
        print("  Next: upload this file to Colab, run colab_embed_diseases.ipynb (GPU),")
        print("  download disease_embeddings.parquet into data/processed/, then run import_embeddings_from_parquet.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
