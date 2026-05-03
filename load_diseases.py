import pandas as pd
import glob
from neo4j import GraphDatabase

# 1. Setup Connection
URI = "bolt://127.0.0.1:7687"
AUTH = ("neo4j", "12345678")
driver = GraphDatabase.driver(URI, auth=AUTH)

# 2. Target the diseases folder
# Adjust this path if your data is located elsewhere
disease_files = glob.glob("D:/LAW MOCK/data/diseases/*.parquet")

def patch_disease_names():
    if not disease_files:
        print("[FATAL] Could not find the 'diseases' folder. Check your path.")
        return

    with driver.session() as session:
        for file in disease_files:
            print(f"Reading dictionary: {file}...")
            # Read only the essential columns to save RAM
            df = pd.read_parquet(file, columns=['id', 'name'])
            df = df.dropna()
            
            # Stringify to prevent protocol crashes
            df['id'] = df['id'].astype(str)
            df['name'] = df['name'].astype(str)
            
            batch = df.to_dict('records')
            
            # The Cypher Injection Query
            query = """
            UNWIND $batch AS row
            // Match the existing node you created last night
            MERGE (d:Disease {id: row.id})
            // Inject the human-readable text
            SET d.name = row.name
            """
            session.run(query, batch=batch)
            print(f"[SUCCESS] Injected names from {file}")

if __name__ == "__main__":
    patch_disease_names()
    print("Database patched. You are cleared to run vectorize.py.")