import glob

import pandas as pd

from src.database.neo4j_client import get_driver, session_kwargs

driver = get_driver()

disease_files = glob.glob("data/diseases/*.parquet")


def patch_disease_names():
    if not disease_files:
        print("[FATAL] Could not find the 'diseases' folder. Check your path.")
        return

    with driver.session(**session_kwargs()) as session:
        for file in disease_files:
            print(f"Reading dictionary: {file}...")
            df = pd.read_parquet(file, columns=["id", "name"])
            df = df.dropna()

            df["id"] = df["id"].astype(str)
            df["name"] = df["name"].astype(str)

            batch = df.to_dict("records")

            query = """
            UNWIND $batch AS row
            MERGE (d:Disease {id: row.id})
            SET d.name = row.name
            """
            session.run(query, batch=batch)
            print(f"[SUCCESS] Injected names from {file}")


if __name__ == "__main__":
    patch_disease_names()
    print("Database patched. You are cleared to run vectorize.py.")
