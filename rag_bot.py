import os
import json
import re
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

# Master AI Config
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_AUTH = ("neo4j", "12345678")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class MedicalGraphRAG:
    def __init__(self):
        print("[SYSTEM] Booting Self-Evaluating Agentic GraphRAG...")
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.strict_debug_mode = os.getenv("STRICT_DEBUG_MODE", "false").lower() in {"1", "true", "yes", "on"}

    def _debug_dump(self, label: str, payload):
        if not self.strict_debug_mode:
            return
        print(f"\n[STRICT DEBUG] {label}")
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        
    def semantic_vector_search(self, query: str):
        print(f"[RAG] Phase 1: Extracting entity & Vectorizing query...")
        
        # Lightweight LLM call to extract Disease entity
        extraction_prompt = "You are a rigid medical entity extractor. Extract ONLY the primary Disease or Gene entity from the user query. Do not include any conversational text. If no entity is found, output exactly NONE."
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": extraction_prompt},
                {"role": "user", "content": f"Query: {query}"}
            ],
            temperature=0.0
        )
        entity = response.choices[0].message.content.strip()
        print(f"         -> Extracted Entity: '{entity}'")
        
        query_vector = self.embedder.encode(entity).tolist()
        cypher = """
        CALL db.index.vector.queryNodes('disease_name_embeddings', 10, $embedding)
        YIELD node, score RETURN node.id AS id, node.name AS name, score
        """
        with self.driver.session() as session:
            res = session.run(cypher, embedding=query_vector).data()
            
        print(f"DEBUG: Vector search found: {res}")
        
        if not res and entity != "NONE":
            print(f"DEBUG: Vector search returned 0 results. Triggering fallback exact/fuzzy match for '{entity}'...")
            fallback_cypher = """
            MATCH (d:Disease)
            WHERE toLower(d.name) CONTAINS toLower($entity_param)
            RETURN d.id AS id, d.name AS name, 1.0 AS score
            LIMIT 10
            """
            with self.driver.session() as session:
                res = session.run(fallback_cypher, entity_param=entity).data()
            print(f"DEBUG: Fallback search found: {res}")
            
        self._debug_dump("Phase 1 Raw Vector Results", res)
        return (res[0]['id'], res[0]['name'], entity) if res else (None, None, entity)

    def deterministic_traversal(self, disease_id: str):
        print(f"[RAG] Phase 2: Extracting topological evidence...")
        
        # Notice the undirected relationship: -[:HAS_EVIDENCE]- (No arrow)
        cypher = """
        MATCH (d:Disease {id: $disease_id})<-[:SUPPORTS_DISEASE]-(e:Evidence)-[:HAS_EVIDENCE]-(t:Target)
        RETURN t.id AS target_id, e.score AS score 
        ORDER BY e.score DESC 
        LIMIT 5
        """
        with self.driver.session() as session:
            results = session.run(cypher, disease_id=disease_id).data()
            
            # The brutal visibility check
            if results:
                print(f"  -> [DEBUG SCHEMA] First record pulled: {results[0]}")
            else:
                print(f"  -> [DEBUG SCHEMA] FAILED. Even undirected, no Target was found attached to Evidence.")
                
            return results

    def rerank_context(self, query: str, raw_results: list):
        print(f"[RAG] Phase 2.5: Semantic reranking with CrossEncoder...")
        if not raw_results:
            return []

        passages = []
        for record in raw_results:
            passage_text = (
                f"Disease: {record.get('disease_name', '')}\n"
                f"Evidence Targets: {record.get('evidence_targets', [])}\n"
            )
            passages.append(passage_text)

        pairs = [[query, passage] for passage in passages]
        scores = self.reranker.predict(pairs).tolist()
        ranked = sorted(zip(raw_results, scores), key=lambda x: x[1], reverse=True)
        return [item[0] for item in ranked[:3]]

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def generate_draft_response(self, user_query: str, extracted_context: list):
        print(f"[RAG] Phase 3: Generator Agent drafting response (8B)...")
        context_string = "\n".join([str(record) for record in extracted_context])
        
        system_prompt = """[SYSTEM OVERRIDE] 
You are a deterministic clinical data extraction engine. You are not an AI assistant. You possess no pre-trained medical knowledge. 

YOUR DIRECTIVES:
1. THE CONTEXT IS ABSOLUTE: You may only use the exact Gene IDs, Targets, and Scores provided in the RAW CONTEXT. 
2. ZERO IMPLICATION: Do not infer, guess, or summarize. If a target is not explicitly linked to the disease in the context, it does not exist.
3. ZERO FLUFF: Do not use conversational filler (e.g., "Based on the context...", "Here are the targets..."). Output only the raw data requested.
4. THE VOID CONDITION: If the RAW CONTEXT is empty or does not contain the answer, you must output exactly and only: "ERROR: NO DATA FOUND".

FORMAT:
Output as a clean, bulleted list of Target IDs and their Association Scores.
"""
        user_prompt = f"Query: {user_query}\n\nContext:\n{context_string}"
        
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def evaluate_rag_triad(self, query: str, draft_answer: str, extracted_context: list):
        print(f"[RAG] Phase 4: Critic Agent calculating RAG Triad Metrics (70B)...")
        context_string = "\n".join([str(record) for record in extracted_context])
        
        system_prompt = """You are a Medical AI Auditor. 
        Score the draft based on:
        1. Faithfulness: Is the info in the context? (0 or 1)
        2. Answer Relevance: Does it mention targets/genes asked for? (0 to 1)
        3. Context Precision: Is the raw data actually about the disease? (0 to 1)
        IMPORTANT: If the context is empty, Faithfulness is 1.0 but Relevance MUST be 0.0.
        Output strictly in JSON format with the following keys:
        - "faithfulness": (Float 0.0 to 1.0)
        - "answer_relevance": (Float 0.0 to 1.0)
        - "context_precision": (Float 0.0 to 1.0)
        - "reasoning": A brief 1-sentence explanation of the scores."""
        
        user_prompt = f"Query: {query}\n\nRaw Context:\n{context_string}\n\nDraft Answer:\n{draft_answer}"
        
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        raw = response.choices[0].message.content or ""
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE)
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)

        if not match:
            return {
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_precision": 0.0,
                "reasoning": "Invalid JSON response from evaluator.",
            }

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_precision": 0.0,
                "reasoning": "JSON parsing failed after regex extraction.",
            }

if __name__ == "__main__":
    bot = MedicalGraphRAG()
    
    user_question = "What are the primary targets for breast cancer?"
    print(f"\n[USER QUERY] {user_question}")
    print("-" * 60)
    
    d_id, d_name, extracted_entity = bot.semantic_vector_search(user_question)
    
    if d_id:
        raw_context = bot.deterministic_traversal(d_id)
        context = bot.rerank_context(user_question, raw_context)
        print(f"DEBUG: Retrieved {len(context)} nodes for entity '{extracted_entity}'")
        draft = bot.generate_draft_response(user_question, context)
        metrics = bot.evaluate_rag_triad(user_question, draft, context)
        
        print("\n[PIPELINE TELEMETRY: THE RAG TRIAD]")
        print("=" * 60)
        print(f"► Faithfulness (Hallucination Check): {metrics['faithfulness']:.2f} / 1.00")
        print(f"► Answer Relevance:                   {metrics['answer_relevance']:.2f} / 1.00")
        print(f"► Context Precision:                  {metrics['context_precision']:.2f} / 1.00")
        print(f"► Critic Reasoning: {metrics['reasoning']}")
        print("=" * 60)
        
        # Strict Enterprise Governance: Must pass all three triad metrics
        faithfulness = float(metrics.get('faithfulness', 0))
        relevance = float(metrics.get('answer_relevance', 0))
        precision = float(metrics.get('context_precision', 0))

        if faithfulness < 1.0:
            print("\n[CRITICAL ERROR] Hallucination detected. Draft dumped.")
        elif relevance < 0.5:
            print("\n[WARNING] Draft is irrelevant to the user query. Draft dumped.")
        elif precision < 0.5:
            print("\n[WARNING] Poor context retrieved. Draft dumped.")
        else:
            print("\n[VALIDATED OUTPUT]")
            print(draft)