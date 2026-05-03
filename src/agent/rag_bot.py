import json
import os

from groq import Groq
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_AUTH = ("neo4j", "12345678")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class MedicalGraphRAG:
    def __init__(self):
        print("[SYSTEM] Booting Self-Evaluating Agentic GraphRAG...")
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = Groq(api_key=GROQ_API_KEY)

    def semantic_vector_search(self, query: str):
        print("[RAG] Phase 1: Vectorizing query & searching 3.7M nodes...")
        query_vector = self.embedder.encode(query).tolist()
        cypher = """
        CALL db.index.vector.queryNodes('disease_name_embeddings', 1, $embedding)
        YIELD node, score RETURN node.id AS id, node.name AS name
        """
        with self.driver.session() as session:
            res = session.run(cypher, embedding=query_vector).data()
        return (res[0]["id"], res[0]["name"]) if res else (None, None)

    def deterministic_traversal(self, disease_id: str):
        print("[RAG] Phase 2: Extracting topological evidence...")
        cypher = """
        MATCH (d:Disease {id: $disease_id})<-[:SUPPORTS_DISEASE]-(e:Evidence)
        RETURN e.targetId AS target, e.score AS score ORDER BY e.score DESC LIMIT 5
        """
        with self.driver.session() as session:
            return session.run(cypher, disease_id=disease_id).data()

    def generate_draft_response(self, user_query: str, extracted_context: list):
        print("[RAG] Phase 3: Generator Agent drafting response (8B)...")
        context_string = "\n".join([str(record) for record in extracted_context])

        system_prompt = "You are a data extraction assistant. Summarize the provided context strictly and concisely."
        user_prompt = f"Query: {user_query}\n\nContext:\n{context_string}"

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content

    def evaluate_rag_triad(self, query: str, draft_answer: str, extracted_context: list):
        print("[RAG] Phase 4: Critic Agent calculating RAG Triad Metrics (70B)...")
        context_string = "\n".join([str(record) for record in extracted_context])

        system_prompt = """You are an Enterprise AI Evaluator. Your job is to score a RAG pipeline.
        You must evaluate the Draft Answer against the Query and the Raw Context.
        Output strictly in JSON format with the following keys:
        - "faithfulness": (Float 0.0 to 1.0) 1.0 if the draft contains ONLY facts from the context. 0.0 if there are hallucinations.
        - "answer_relevance": (Float 0.0 to 1.0) 1.0 if the draft directly answers the user query.
        - "context_precision": (Float 0.0 to 1.0) 1.0 if the context provided is highly relevant to the query.
        - "reasoning": A brief 1-sentence explanation of the scores."""

        user_prompt = f"Query: {query}\n\nRaw Context:\n{context_string}\n\nDraft Answer:\n{draft_answer}"

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return json.loads(response.choices[0].message.content)


if __name__ == "__main__":
    bot = MedicalGraphRAG()

    user_question = "What are the primary targets for breast cancer?"
    print(f"\n[USER QUERY] {user_question}")
    print("-" * 60)

    d_id, d_name = bot.semantic_vector_search(user_question)

    if d_id:
        context = bot.deterministic_traversal(d_id)
        draft = bot.generate_draft_response(user_question, context)
        metrics = bot.evaluate_rag_triad(user_question, draft, context)

        print("\n[PIPELINE TELEMETRY: THE RAG TRIAD]")
        print("=" * 60)
        print(f"-> Faithfulness (Hallucination Check): {metrics['faithfulness']:.2f} / 1.00")
        print(f"-> Answer Relevance:                   {metrics['answer_relevance']:.2f} / 1.00")
        print(f"-> Context Precision:                  {metrics['context_precision']:.2f} / 1.00")
        print(f"-> Critic Reasoning: {metrics['reasoning']}")
        print("=" * 60)

        if float(metrics["faithfulness"]) < 1.0:
            print("\n[CRITICAL ERROR] Hallucination detected. Draft dumped.")
        else:
            print("\n[VALIDATED OUTPUT]")
            print(draft)
