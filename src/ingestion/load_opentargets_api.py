import sys
import logging
import requests
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.database.neo4j_client import get_driver, session_kwargs

logging.basicConfig(level=logging.INFO, format="%(asctime)s INFO - %(message)s")
logger = logging.getLogger("api_ingest")

API_URL = "https://api.platform.opentargets.org/api/v4/graphql"

def fetch_disease_ids(driver, limit=500):
    """Fetch random EFO diseases to bulk populate the graph with targets."""
    query = """
    MATCH (d:Disease)
    WHERE NOT (d)-[:HAS_TARGET]->()
      AND d.id STARTS WITH 'EFO_'
    WITH d, rand() AS r
    ORDER BY r
    RETURN d.id AS id
    LIMIT $limit
    """
    with driver.session(**session_kwargs()) as session:
        result = session.run(query, limit=limit)
        return [record["id"] for record in result]

def get_targets_from_api(efo_id: str):
    """Ping OpenTargets GraphQL with the correct efoId parameter."""
    query = """
    query getTargets($efoId: String!) {
      disease(efoId: $efoId) {
        associatedTargets(page: {index: 0, size: 10}) {
          rows {
            target { id, approvedSymbol }
            score
          }
        }
      }
    }
    """
    response = requests.post(API_URL, json={"query": query, "variables": {"efoId": efo_id}})
    if response.status_code == 200:
        data = response.json()
        
        # If the API throws a GraphQL error, log it so we don't fail silently again
        if "errors" in data:
            logger.error(f"GraphQL Error for {efo_id}: {data['errors'][0]['message']}")
            return []
            
        try:
            return data["data"]["disease"]["associatedTargets"]["rows"]
        except (KeyError, TypeError):
            return []
    return []

def write_edges_to_neo4j(driver, disease_id: str, targets: list):
    if not targets:
        return
    
    query = """
    UNWIND $targets AS t
    MERGE (tgt:Target {id: t.target.id})
    SET tgt.symbol = t.target.approvedSymbol
    MERGE (d:Disease {id: $disease_id})
    MERGE (d)-[r:HAS_TARGET]->(tgt)
    SET r.score = t.score
    """
    with driver.session(**session_kwargs()) as session:
        session.run(query, disease_id=disease_id, targets=targets).consume()

def main():
    load_dotenv(ROOT / ".env")
    driver = get_driver()
    
    try:
        logger.info("Fetching disease IDs from Neo4j...")
        disease_ids = fetch_disease_ids(driver, limit=100) 
        
        if not disease_ids:
            logger.info("No isolated diseases found. Graph is fully connected.")
            return

        logger.info(f"Processing {len(disease_ids)} diseases via OpenTargets API...")
        
        for idx, efo_id in enumerate(disease_ids, 1):
            targets = get_targets_from_api(efo_id)
            if targets:
                write_edges_to_neo4j(driver, efo_id, targets)
                logger.info(f"[{idx}/{len(disease_ids)}] Linked {len(targets)} targets to {efo_id}")
            else:
                logger.info(f"[{idx}/{len(disease_ids)}] No targets found for {efo_id}")
                
        logger.info("Batch ingestion complete. Your graph now has edges.")
    finally:
        driver.close()

if __name__ == "__main__":
    main()