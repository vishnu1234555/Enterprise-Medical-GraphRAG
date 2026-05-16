"""
Run Neo4j diagnostics, then disease dictionary download → name patch → vectorize.
Evidence parquet ingestion is separate (not in this repo).

Usage (from project root):
    python run_medical_ingest.py
    python run_medical_ingest.py --colab-prep
        Same as above but skips local CPU vectorize; exports Parquet for Colab GPU
        (see notebooks/colab_embed_diseases.ipynb), then you import embeddings locally.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--colab-prep",
        action="store_true",
        help="Export diseases_to_embed.parquet for Colab T4; skip local vectorize.py",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

    from src.database.neo4j_client import get_driver, session_kwargs

    print("\n" + "=" * 60)
    print("STEP 0 — Neo4j diagnostics (this app’s .env connection)")
    print("=" * 60)
    driver = get_driver()
    try:
        with driver.session(**session_kwargs()) as session:
            diseases = session.run(
                "MATCH (d:Disease) RETURN count(d) AS c"
            ).single()["c"]
            named = session.run(
                "MATCH (d:Disease) WHERE d.name IS NOT NULL "
                "AND trim(toString(d.name)) <> '' RETURN count(d) AS c"
            ).single()["c"]
            evidence = session.run(
                "MATCH (:Evidence) RETURN count(*) AS c"
            ).single()["c"]
            try:
                idx = session.run(
                    "SHOW INDEXES WHERE name = 'disease_name_embeddings' RETURN name LIMIT 1"
                ).data()
            except Exception:
                idx = []
        print(f"  Disease nodes (total):     {diseases}")
        print(f"  Disease with `name` set:   {named}")
        print(f"  Evidence nodes:            {evidence}")
        print(
            "  Vector index present:      "
            + ("yes" if idx else "no (run vectorize.py after names)")
        )
    except Exception as exc:
        print(f"  ERROR: {exc}")
        print("  Fix NEO4J_URI / credentials in .env and retry.")
        sys.exit(1)
    finally:
        driver.close()

    steps = [
        ("STEP 1 — Download OpenTargets disease parquet (FTP)", "src.ingestion.fetch_diseases"),
        ("STEP 2 — Merge disease names into Neo4j", "src.ingestion.load_diseases"),
    ]
    if args.colab_prep:
        steps.append(
            (
                "STEP 3 — Export Parquet for Colab GPU embedding",
                "src.ingestion.export_diseases_for_colab",
            )
        )
    else:
        steps.append(
            (
                "STEP 3 — Build vector index + embeddings (local CPU)",
                "src.ingestion.vectorize",
            )
        )

    for title, mod in steps:
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)
        r = subprocess.run(
            [sys.executable, "-m", mod],
            cwd=str(ROOT),
        )
        if r.returncode != 0:
            print(f"\n[WARN] Module {mod} exited with code {r.returncode}")
            if mod == "src.ingestion.fetch_diseases":
                print("  (If FTP failed, place *.parquet under data/diseases/ and re-run.)")

    print("\n" + "=" * 60)
    if args.colab_prep:
        print("Colab prep done.")
        print("  1. Upload data/processed/diseases_to_embed.parquet to Colab.")
        print("  2. Open notebooks/colab_embed_diseases.ipynb (Runtime → GPU).")
        print("  3. Download disease_embeddings.parquet → data/processed/")
        print("  4. Run: python -m src.ingestion.import_embeddings_from_parquet")
    else:
        print("Done. Restart Streamlit or use “Reconnect Neo4j” in the sidebar.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()