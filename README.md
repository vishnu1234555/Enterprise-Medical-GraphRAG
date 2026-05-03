# 🧬 Enterprise Medical GraphRAG

![Architecture](https://img.shields.io/badge/Architecture-GraphRAG-blue)
![Database](https://img.shields.io/badge/Database-Neo4j_3.8M_Nodes-0482dc)
![LLM](https://img.shields.io/badge/LLM-Llama_3_70B_%7C_8B-black)
![UI](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B)

## 📌 The Architecture
Most AI wrappers hallucinate in production because they lack deterministic guardrails. This system is a **Self-Evaluating Medical GraphRAG** designed to completely eliminate LLM drift through structural graph traversal and agentic auditing.

The pipeline executes a strict 4-phase retrieval protocol:

1. **Semantic Vector Search:** User queries are vectorized (`all-MiniLM-L6-v2`) and matched against disease nodes with a hard **85% confidence threshold**. Irrelevant/adversarial queries are dumped at the vector level, saving compute.
2. **Deterministic Topological Traversal:** Bypassing standard semantic chunking, the system traverses a 3.8M node Neo4j graph using the undirected schema: `(Disease)<-[:SUPPORTS_DISEASE]-(Evidence)-[:HAS_EVIDENCE]-(Target)`.
3. **Anti-Gravity Generator (8B):** An 8B parameter model drafts the response using a strict "Anti-Gravity" prompt that strips conversational filler and forces a `ERROR: NO DATA FOUND` void condition if the graph context is empty.
4. **The Critic Agent (70B):** A 70B auditor evaluates the draft against the raw graph data, scoring the **RAG Triad** (Faithfulness, Relevance, Context Precision). Any hallucination results in an immediate draft rejection.

## 🚀 Key Engineering Features
* **Zero-Cost Adversarial Defense:** Non-medical queries ("How to cook rice", SQL injections) fail the vector threshold and are rejected before LLM inference.
* **Memory-Optimized Ingestion:** Data pipeline utilizes chunked batching and Cypher `UNWIND` queries to bypass Java Heap limits during massive OpenTargets Parquet ingestion.
* **Real-time Telemetry UI:** Streamlit frontend streams the drafting process and displays live RAG Triad scores.

## 🛠️ Quickstart (Docker Deployment)

You do not need local Python dependencies to run this application. It is fully containerized.

**1. Clone the repository**
```bash
git clone https://github.com/vishnu1234555/Enterprise-Medical-GraphRAG.git
cd Enterprise-Medical-GraphRAG
```

**2. Configure Environment Variables**
Create a `.env` file in the root directory:
```bash
GROQ_API_KEY=gsk_your_key_here
NEO4J_URI=bolt://host.docker.internal:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

**3. Build and Run the Container**
```bash
docker build -t enterprise-medical-rag .
docker run -p 8501:8501 --env-file .env enterprise-medical-rag
```
