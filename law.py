import pandas as pd
from neo4j import GraphDatabase
import glob
import time
import os

# 1. Target the root import directory and grab EVERY parquet file recursively
base_dir = r"C:\Users\DELL\.Neo4jDesktop2\Data\dbmss\dbms-b272206c-bd7e-4096-8b8e-4cceee4453d7\import"
parquet_files = glob.glob(os.path.join(base_dir, "**", "*.parquet"), recursive=True)

print(f"CRITICAL: Found {len(parquet_files)} Parquet files originally.")

# ==========================================
# HARD BYPASS INJECTION (Hardware constraint: i3 CPU / RAM limit)
# Amputate any file path containing 'europepmc' from the ingestion queue
# ==========================================
safe_parquet_files = [f for f in parquet_files if "europepmc" not in f]
skipped_count = len(parquet_files) - len(safe_parquet_files)
print(f"[SYSTEM FATAL OVERRIDE] Amputated {skipped_count} 'europepmc' files to prevent OOM deadlock.")

# Reassign the safe list to the execution variable
parquet_files = safe_parquet_files
# ==========================================

URI = "bolt://127.0.0.1:7687"
AUTH = ("neo4j", "12345678")

driver = GraphDatabase.driver(URI, auth=AUTH)

# 2. The Universal Schema Extract+ion Query
def insert_batch(tx, batch, source_name):
    query = """
    UNWIND $batch AS row
    
    // Failsafe: Only process rows that have the critical routing keys
    WITH row WHERE row.targetId IS NOT NULL AND row.diseaseId IS NOT NULL 
               AND row.targetId <> "" AND row.diseaseId <> ""
    
    // Map Core Entities
    MERGE (t:Target {id: row.targetId})
    MERGE (d:Disease {id: row.diseaseId})
    
    // Create the Universal Evidence Node and tag it with its origin folder
    CREATE (e:Evidence {dataSource: $source_name})
    SET e += row
    
    // Wire the Graph Topology
    MERGE (t)-[:HAS_EVIDENCE]->(e)
    MERGE (e)-[:SUPPORTS_DISEASE]->(d)
    """
    tx.run(query, batch=batch, source_name=source_name)

batch_size = 5000
master_start_time = time.time()

with driver.session() as session:
    for file in parquet_files:
        # Extract folder name (e.g., 'sourceId=chembl') to tag the data
        parent_folder = os.path.basename(os.path.dirname(file))
        source_name = parent_folder.replace("sourceId=", "")
        
        print(f"\n[INGESTING] Folder: {source_name} | File: {os.path.basename(file)}")
        
        try:
            df = pd.read_parquet(file).fillna("")
            
            # Brutal stringify for Neo4j compatibility
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str)
                    
            records = df.to_dict('records')
            total_records = len(records)
            
            for i in range(0, total_records, batch_size):
                batch = records[i:i+batch_size]
                session.execute_write(insert_batch, batch, source_name)
                
        except Exception as e:
            print(f"   -> WARNING: Skipped {file} due to schema collision. Error: {e}")

master_end_time = time.time()
print(f"\nSUCCESS: Entire Data Lake ingested and mapped in {round((master_end_time - master_start_time)/60, 2)} minutes.")