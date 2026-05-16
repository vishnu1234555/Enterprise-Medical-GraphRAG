"""
src/agent/rag_bot.py
====================
Medical GraphRAG Agent

Pipeline:
  1. RETRIEVE  — Semantic vector search to find the best-matching Disease node.
  2. TRAVERSE  — Deterministic Cypher traversal to pull Evidence → Target facts.
  3. ACTOR     — LLM drafts an initial answer grounded in the raw graph context.

No external modules beyond those already compiled in the Docker image are used.
"""

import logging
import os
import re
import json
import urllib.request
from typing import Any

from groq import AsyncGroq
from neo4j.exceptions import Neo4jError
from tenacity import retry, stop_after_attempt, wait_exponential

from src.agent.prompts import (
    ROUTER_SYSTEM,
    ACTOR_SYSTEM,
    ACTOR_USER
)
from src.utils.text_processing import (
    build_context,
    build_target_context,
    format_chat_history,
    history_section,
    format_retrieval_diagnostics
)
from src.database.neo4j_queries import (
    execute_target_search,
    execute_target_traversal,
    execute_lexical_disease_search,
    execute_id_bypass_search,
    execute_vector_search,
    execute_disease_traversal,
    get_retrieval_diagnostics_stats
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_LEX_STOP = frozenset(
    "a an the and or for to of in on at by as is are was were be been being "
    "it this that these those what which who how when where why with from "
    "into about over under than then not no yes do does did doing done can "
    "could should would will just only also very more most some any each "
    "all both explain describe tell give list key primary main targets gene "
    "therapy disease condition patient clinical".split()
)

def query_tokens(query: str) -> list[str]:
    raw = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{2,}", query.lower())
    return [t for t in raw if t not in _LEX_STOP]

def fallback_lexical_search(driver, query: str) -> tuple[str | None, str | None, str | None]:
    qstrip = query.strip()
    if not qstrip:
        return None, None, None
    tokens = query_tokens(qstrip)
    qlower = qstrip.lower()
    
    records = execute_lexical_disease_search(driver, qlower, tokens)
    
    if not records:
        log.warning("Lexical fallback found no Disease rows for query: %r", query)
        return None, None, None

    def score_row(rec: dict[str, Any]) -> tuple[int, int]:
        name = str(rec.get("name") or "").lower()
        did = str(rec.get("id") or "").lower()
        blob = f"{name} {did}"
        if tokens:
            hits = sum(1 for t in tokens if t and t in blob)
        else:
            hits = 1 if (qlower in name or qlower in did) else 0
        return (hits, len(blob))

    best = max(records, key=score_row)
    log.info("Lexical fallback → Disease: '%s' (id=%s)", best.get("name"), best.get("id"))
    return best.get("id"), best.get("name"), None

def pick_best_vector_hit(query: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    qlower = query.strip().lower()
    for row in records:
        name = str(row.get("name") or "").lower()
        if name and qlower in name:
            return row
    for row in records:
        name = str(row.get("name") or "").lower()
        if any(t in name for t in query_tokens(query)):
            return row
    return records[0]

def semantic_vector_search(driver, embedder, query: str, vector_index: str, top_k: int) -> tuple[str | None, str | None, str | None]:
    if not query or not query.strip():
        return None, None, None

    match = re.search(r'(EFO_\d+|MONDO_\d+|DOID_\d+|GO_\d+)', query)
    if match:
        extracted_id = match.group(1)
        records = execute_id_bypass_search(driver, extracted_id)
        
        if records and records[0].get("id"):
            disease_name = records[0].get("name") or "Unknown"
            targets = records[0].get("targets") or []
            disease_id = records[0].get("id")
            targets_clean = [str(t) for t in targets if t]
            
            context = f"Disease Entity: {disease_name} (ID: {disease_id})\n\nSupporting Evidence (Target | Score | Source):\n---------------------------------------------\n"
            if targets_clean:
                for t in targets_clean:
                    t_name = _translate_chembl(t)
                    context += f"  • {t_name:<30} | score: 1.0000 | source: Exact ID Match\n"
            else:
                context += f"  • No targets found for this exact ID match.\n"
            
            log.info("ID Bypass → Disease: '%s' (id=%s)", disease_name, extracted_id)
            return extracted_id, disease_name, context
        else:
            log.warning("ID Bypass found no Disease for ID: %s", extracted_id)
            return None, None, None

    cleaned = query.strip()
    embedding = embedder.encode(cleaned).tolist()

    try:
        records = execute_vector_search(driver, vector_index, top_k, embedding)
    except Neo4jError as exc:
        log.warning("Vector index query failed (%s); using lexical fallback.", exc)
        return fallback_lexical_search(driver, cleaned)

    if not records:
        log.warning("Vector search returned no results; trying lexical fallback.")
        return fallback_lexical_search(driver, cleaned)

    top = pick_best_vector_hit(cleaned, records)
    log.info(
        "Vector search → Disease: '%s' (id=%s, score=%s)",
        top.get("name"),
        top.get("id"),
        top.get("score", "n/a"),
    )
    return top.get("id"), top.get("name"), None

_CHEMBL_CACHE = {}

def _translate_chembl(chembl_id: str) -> str:
    """Instantly translates raw ChEMBL IDs into human-readable names via the EBI API."""
    if not isinstance(chembl_id, str) or not chembl_id.startswith("CHEMBL"):
        return chembl_id
    if chembl_id in _CHEMBL_CACHE:
        return _CHEMBL_CACHE[chembl_id]
        
    try:
        # Check the Target database first
        url = f"https://www.ebi.ac.uk/chembl/api/data/target/{chembl_id}.json"
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=1.5) as response:
            data = json.loads(response.read().decode())
            name = data.get('pref_name', chembl_id)
            _CHEMBL_CACHE[chembl_id] = name
            return name
    except Exception:
        try:
            # Fallback to the Molecule database
            url_mol = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
            req = urllib.request.Request(url_mol, headers={'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=1.5) as response:
                data = json.loads(response.read().decode())
                name = data.get('pref_name', chembl_id)
                _CHEMBL_CACHE[chembl_id] = name
                return name
        except Exception:
            _CHEMBL_CACHE[chembl_id] = chembl_id
            return chembl_id

class MedicalAgent:
    """
    Medical GraphRAG agent.

    Public API
    ----------
    execute_query(user_query: str, chat_history: list | None = None) -> str
        Full pipeline: retrieve → draft → return.

    process_query(user_query: str, chat_history: list | None = None) -> str
        Alias kept for backward compatibility with app.py.
    """
    _MAX_EVIDENCE = 5
    _VECTOR_INDEX = "disease_name_embeddings"
    _TOP_K = 8

    def __init__(self, driver, embedder) -> None:
        log.info("Booting MedicalAgent …")
        self.driver = driver
        self.embedder = embedder
        log.info("Driver and Embedder injected.")
        self.groq = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        self.actor_model = os.getenv("GROQ_ACTOR_MODEL", "llama-3.3-70b-versatile")
        log.info("Groq client ready — actor_model: %s", self.actor_model)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _route_query(self, query: str) -> dict:
        """Traffic cop: decides if we search the Disease index or Target index."""
        router_model = os.getenv("GROQ_ROUTER_MODEL", "llama-3.1-8b-instant")
        response = await self.groq.chat.completions.create(
            model=router_model,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            response_format={"type": "json_object"} # Force JSON output
        )
        
        try:
            raw_content = response.choices[0].message.content
            clean_content = raw_content.replace('```json', '').replace('```', '').strip()
            return json.loads(clean_content)
        except json.JSONDecodeError:
            return {"intent": "disease", "entity": query} # Fallback

    def _query_tokens(self, query: str) -> list[str]:
        return query_tokens(query)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _draft_response(
        self, query: str, context: str, conversation_history: str
    ) -> str:
        """Generate an initial answer (Actor step)."""
        log.info("Actor: drafting response …")
        hist_section = history_section(conversation_history)
        response = await self.groq.chat.completions.create(
            model=self.actor_model,
            messages=[
                {"role": "system", "content": ACTOR_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"{hist_section}"
                        f"<GRAPH_CONTEXT>\n{context}\n</GRAPH_CONTEXT>\n\n"
                        f"User Question: {query}\n\n"
                        f"{ACTOR_USER}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        draft = response.choices[0].message.content
        log.info("Actor draft complete (%d chars).", len(draft))
        return draft

    async def execute_query(self, user_query: str, chat_history: list | None = None) -> str:
        """
        Full RAG pipeline.
        """
        conversation_history = format_chat_history(chat_history)

        cleaned_query = user_query.strip()
        if not cleaned_query:
            return (
                "Please enter a valid medical question — e.g., "
                "*'What are the primary targets for breast cancer?'*"
            )

        if not os.getenv("GROQ_API_KEY"):
            return (
                "⚠️ **`GROQ_API_KEY` is not set.** Add it to your `.env` in the project root, "
                "then use the sidebar **Reconnect Neo4j (reload agent)** or restart Streamlit."
            )

        try:
            routing_data = await self._route_query(cleaned_query)
            intent = routing_data.get("intent")
            entity = routing_data.get("entity", cleaned_query)
            
            if intent == "conversational":
                draft = await self._draft_response(cleaned_query, "No clinical context needed.", conversation_history)
                return draft
                
            elif intent == "target":
                qlower = entity.strip().lower()
                target_records = execute_target_search(self.driver, qlower)
                if not target_records:
                    return f"Could not find a Target matching '{entity}' in the database."
                target_id, target_name = target_records[0]["id"], target_records[0]["name"]
                target_name = _translate_chembl(target_name)
                
                evidence = execute_target_traversal(self.driver, target_id, self._MAX_EVIDENCE)
                context = build_target_context(target_name, evidence)
                
            else:
                disease_id, disease_name, direct_context = semantic_vector_search(
                    self.driver, self.embedder, entity, self._VECTOR_INDEX, self._TOP_K
                )
    
                if not disease_id:
                    stats = get_retrieval_diagnostics_stats(self.driver, self._VECTOR_INDEX)
                    diag = format_retrieval_diagnostics(stats, self._VECTOR_INDEX)
                    return (
                        "I couldn't match your question to a **`Disease`** node in Neo4j "
                        "(vector index + name/id fallback both returned nothing).\n\n"
                        "**What to do:**\n"
                        "1. In the sidebar, click **Reconnect Neo4j (reload agent)** after any `.env` change.\n"
                        "2. Confirm data: run `MATCH (d:Disease) RETURN count(d)` in Neo4j Browser.\n"
                        "3. Ingest disease **names** (`load_diseases.py`), then run **`vectorize.py`** for embeddings.\n\n"
                        "**Live diagnostics from this app’s Neo4j session:**\n"
                        f"{diag}"
                    )
    
                if direct_context:
                    context = direct_context
                else:
                    evidence = execute_disease_traversal(self.driver, disease_id, self._MAX_EVIDENCE)
                    log.info("Graph traversal → %d evidence records found.", len(evidence))
                    unique_evidence = []
                    seen_targets = set()

                    for rec in evidence:
                        raw_target = rec.get("target")
                        if raw_target:
                            translated_target = _translate_chembl(raw_target)
                            rec["target"] = translated_target

                            # Only add to context if we haven't seen this exact target name yet
                            if translated_target not in seen_targets:
                                seen_targets.add(translated_target)
                                unique_evidence.append(rec)

                    evidence = unique_evidence
                    desc = evidence[0].get("description") if evidence else None
                    context = build_context(disease_name, evidence, desc)

            final_answer = await self._draft_response(cleaned_query, context, conversation_history)
            return final_answer

        except Exception as exc:
            log.exception("Pipeline error for query: %r", user_query)
            return (
                f"⚠️ An internal error occurred while reasoning over the knowledge graph:\n\n"
                f"`{type(exc).__name__}: {exc}`\n\n"
                "Please check your Neo4j connection and Groq API key, then retry."
            )

    async def process_query(self, user_query: str, chat_history: list | None = None) -> str:
        return await self.execute_query(user_query, chat_history=chat_history)
