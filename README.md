# Enterprise Medical GraphRAG 🧬

A production-grade, deterministic Clinical AI assistant built to query biomedical knowledge graphs with zero hallucination. 

This architecture leverages a highly optimized Retrieval-Augmented Generation (RAG) pipeline, grounding Large Language Model (LLM) reasoning strictly in peer-reviewed data from the OpenTargets database via Neo4j.

## 🚀 Architecture & Optimization

This system was engineered to run efficiently under strict hardware constraints, offloading heavy vector computation and implementing dynamic API interceptors.

* **Hybrid Retrieval Engine:** Combines dense vector search (`all-MiniLM-L6-v2`) for semantic disease matching with deterministic Cypher graph traversals to extract hard clinical evidence.
* **Anti-Hallucination Constraints:** Strict XML bounding forces the Llama-3 model to separate its internal medical knowledge (disease overviews) from graph-grounded reality (validated targets).
* **Live API Interceptor:** Bypasses local database bloat by intercepting raw `CHEMBL` IDs and dynamically pinging the EBI (European Bioinformatics Institute) API to translate them into human-readable biological targets in milliseconds.
* **Asynchronous Processing:** Built with `asyncio` and `AsyncGroq` for high-concurrency request handling, dropping latency by eliminating sequential blocking.

## 🛠️ Tech Stack

* **Backend Logic:** Pure Python 3.10+ (`asyncio`, `tenacity`)
* **Knowledge Graph:** Neo4j (Cypher, Vector Indexes)
* **LLM Engine:** Groq API (Llama-3 Series)
* **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
* **Frontend:** Streamlit (Persistent chat state, typewriter streaming)

## 🗄️ Ingestion Pipeline (Hardware-Aware)

To handle 30,000+ biological targets without bottlenecking local CPU threads, the ingestion pipeline is decoupled:
1.  **Extract:** FTP fetch of raw OpenTargets Parquet dictionaries.
2.  **Load:** Direct UNWIND injection into Neo4j nodes.
3.  **Embed (Offloaded):** Extracts unembedded nodes to Parquet, offloads 384-dimensional vector encoding to Google Colab (T4 GPU), and re-ingests the optimized vectors back into the local graph.
4.  **Connect:** Pings the OpenTargets GraphQL API to establish `[:HAS_TARGET]` clinical evidence relationships.

## ⚙️ Setup & Execution

1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/vishnu1234555/Enterprise-Medical-GraphRAG.git
   cd Enterprise-Medical-GraphRAG
   pip install -r requirements.txt
   ```

2. Configure your `.env` file:
   ```env
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   GROQ_API_KEY=your_groq_api_key
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```
