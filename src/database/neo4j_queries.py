from typing import Any
from src.database.neo4j_client import session_kwargs

def execute_target_search(driver, qlower: str) -> list[dict]:
    cypher = """
    MATCH (t:Target)
    WHERE toLower(toString(t.id)) CONTAINS $qlower 
       OR toLower(toString(t.symbol)) CONTAINS $qlower
       OR toLower(toString(t.name)) CONTAINS $qlower
    RETURN t.id AS id, coalesce(t.symbol, t.name, t.id) AS name
    LIMIT 1
    """
    with driver.session(**session_kwargs()) as session:
        return session.run(cypher, qlower=qlower).data()

def execute_target_traversal(driver, target_id: str, limit: int) -> list[dict]:
    cypher = """
    MATCH (t:Target {id: $target_id})-[:HAS_EVIDENCE]-(e:Evidence)-[:SUPPORTS_DISEASE]-(d:Disease)
    RETURN d.name AS disease, coalesce(e.score, 1.0) AS score
    ORDER BY score DESC
    LIMIT $limit
    """
    with driver.session(**session_kwargs()) as session:
        return session.run(cypher, target_id=target_id, limit=limit).data()

def execute_lexical_disease_search(driver, qlower: str, tokens: list[str]) -> list[dict]:
    cypher = """
    MATCH (d:Disease)
    WHERE (
        (d.name IS NOT NULL AND trim(toString(d.name)) <> '' AND (toLower(toString(d.name)) CONTAINS $qlower OR ANY(tok IN $tokens WHERE toLower(toString(d.name)) CONTAINS tok)))
        OR ANY(tok IN $tokens WHERE toLower(toString(d.id)) CONTAINS tok)
    )
    RETURN d.id AS id, d.name AS name
    LIMIT 12
    """
    with driver.session(**session_kwargs()) as session:
        return session.run(cypher, qlower=qlower, tokens=tokens or [qlower]).data()

def execute_id_bypass_search(driver, extracted_id: str) -> list[dict]:
    cypher_id = """
    MATCH (d:Disease {id: $extracted_id})
    OPTIONAL MATCH (d)-[:SUPPORTS_DISEASE]-(e:Evidence)
    RETURN d.name AS name, d.id AS id, collect(coalesce(e.targetFromSource, e.targetId)) AS targets
    """
    with driver.session(**session_kwargs()) as session:
        return session.run(cypher_id, extracted_id=extracted_id).data()

def execute_vector_search(driver, index: str, k: int, embedding: list[float]) -> list[dict]:
    cypher = """
    CALL db.index.vector.queryNodes($index, $k, $embedding)
    YIELD node, score
    RETURN node.id AS id, node.name AS name, score
    """
    with driver.session(**session_kwargs()) as session:
        return session.run(cypher, index=index, k=k, embedding=embedding).data()

def execute_disease_traversal(driver, disease_id: str, limit: int) -> list[dict]:
    cypher = """
    MATCH (d:Disease {id: $disease_id})-[r1:SUPPORTS_DISEASE]-(e:Evidence)
    OPTIONAL MATCH (t:Target) WHERE t.id = e.targetId OR t.id = e.targetFromSource
    RETURN 
        coalesce(t.symbol, t.name, e.targetFromSource, e.targetId, 'Unknown Target') AS target,
        coalesce(e.score, 1.0) AS score,
        coalesce(e.dataSource, 'OpenTargets') AS source_info
    ORDER BY score DESC
    LIMIT $limit
    """
    with driver.session(**session_kwargs()) as session:
        return session.run(cypher, disease_id=disease_id, limit=limit).data()

def get_retrieval_diagnostics_stats(driver, vector_index: str) -> dict:
    stats = {}
    try:
        with driver.session(**session_kwargs()) as session:
            stats['total'] = session.run("MATCH (d:Disease) RETURN count(d) AS c").single()["c"]
            stats['named'] = session.run(
                "MATCH (d:Disease) WHERE d.name IS NOT NULL AND trim(toString(d.name)) <> '' "
                "RETURN count(d) AS c"
            ).single()["c"]
            stats['sample'] = session.run(
                "MATCH (d:Disease) RETURN d.id AS id, d.name AS name LIMIT 5"
            ).data()
            try:
                idx = session.run(
                    "SHOW INDEXES WHERE name = $n RETURN name LIMIT 1",
                    n=vector_index,
                ).data()
                stats['idx_visible'] = bool(idx)
            except Exception:
                stats['idx_visible'] = False
    except Exception as e:
        stats['error'] = str(e)
    return stats
