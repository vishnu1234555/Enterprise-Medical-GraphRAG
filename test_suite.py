import time
import json
import traceback

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

from rag_bot import MedicalGraphRAG

# Define the expected 3.8M node schema for reference in tests
SCHEMA_INFO = "(Disease)<-[:SUPPORTS_DISEASE]-(Evidence)-[:HAS_EVIDENCE]->(Target)"

QUERIES = [
    # Direct Medical
    ("Direct Medical", "Primary targets for breast cancer?"),
    ("Direct Medical", "Gene targets for Alzheimer's?"),
    ("Direct Medical", "DHT regulation targets?"),
    ("Direct Medical", "Targeted therapy for non-small cell lung cancer?"),

    # Adversarial/Irrelevant
    ("Adversarial", "How to file a legal petition?"),
    ("Adversarial", "Best way to cook black rice?"),
    ("Adversarial", "Who won the World Cup?"),
    ("Adversarial", "What are the rules of monopoly?"),

    # Edge Case
    ("Edge Case", "Targets for [random gibberish]"),
    ("Edge Case", "Give me targets for cancer"),
    ("Edge Case", " "),
    ("Edge Case", "1=1; DROP TABLE Diseases;")
]

def run_stress_test():
    print("=" * 80)
    print("BOOTING SYSTEMATIC STRESS TEST SUITE")
    print(f"Target Schema Enforcement: {SCHEMA_INFO}")
    print("=" * 80)
    
    try:
        bot = MedicalGraphRAG()
    except Exception as e:
        print(f"FAILED TO INITIALIZE MEDICAL GRAPH RAG PIPELINE: {e}")
        return

    results = []

    for category, query_text in QUERIES:
        print(f"\n[TESTING] Category: {category} | Query: '{query_text}'")
        
        status = "N/A"
        f_val_str = "N/A"
        r_val_str = "N/A"
        p_val_str = "N/A"
        draft_preview = "N/A" # NEW: To capture the text
        
        try:
            # 1. Vector Search
            d_id, d_name, extracted_entity = bot.semantic_vector_search(query_text)
            print(f"  -> [VECTOR RAG] Found Node: '{d_name}' (ID: {d_id})")
            
            if d_id:
                # 2. Graph Traversal
                raw_context = bot.deterministic_traversal(d_id)
                
                # THE SANITY CHECK
                if not raw_context:
                    print(f"  -> [CRITICAL] Graph returned 0 evidence nodes for {d_name}.")
                    status = "DUMPED (Graph Empty)"
                else:
                    print(f"  -> [TRAVERSAL] Extracted {len(raw_context)} nodes. PAYLOAD: {raw_context}")
                    
                    # 3. Reranking (Comment this out if FlashRank isn't fully working yet)
                    # context = bot.rerank_context(query_text, raw_context)
                    context = raw_context 
                    
                    # 4. Drafting
                    draft = bot.generate_draft_response(query_text, context)
                    
                    # THE EXPOSURE LOG
                    print("-" * 60)
                    print(f"  [THE RAW DRAFT]:\n  {draft}")
                    print("-" * 60)
                    
                    # Save a short snippet for the table
                    clean_draft = draft.replace('\n', ' ')
                    draft_preview = clean_draft[:40] + "..." if len(clean_draft) > 40 else clean_draft
                    
                    # 5. Triad Evaluation
                    metrics = bot.evaluate_rag_triad(query_text, draft, context)
                    
                    f_val = float(metrics.get('faithfulness', 0.0))
                    r_val = float(metrics.get('answer_relevance', 0.0))
                    p_val = float(metrics.get('context_precision', 0.0))
                    
                    f_val_str = f"{f_val:.2f}"
                    r_val_str = f"{r_val:.2f}"
                    p_val_str = f"{p_val:.2f}"
                    
                    if f_val < 1.0:
                        status = "DUMPED (Hallucination)"
                    elif r_val < 0.5:
                        status = "DUMPED (Irrelevant)"
                    elif p_val < 0.5:
                        status = "DUMPED (Poor Context)"
                    else:
                        status = "VALIDATED"
            else:
                status = "DUMPED (No Entity Found)"
                
        except Exception as e:
            print(f"  -> [ERROR] Pipeline crashed: {e}")
            status = f"CRASH ({type(e).__name__})"
            
        display_query = query_text[:30] + "..." if len(query_text) > 30 else query_text
        
        # Add the draft preview to the results row
        results.append([
            category,
            display_query,
            f_val_str,
            r_val_str,
            p_val_str,
            draft_preview, # Inject the text here
            status
        ])

    print("\n")
    print("=" * 110)
    print(" MEDICAL GRAPH RAG - STRESS TEST RESULTS ")
    print("=" * 110)
    
    headers = ["Category", "Query", "Faith", "Relev", "Precis", "Draft Output", "Status"]
    
    if HAS_TABULATE:
        print(tabulate(results, headers=headers, tablefmt="grid"))
    else:
        # Fallback to clean string formatting if tabulate is not installed
        row_format = "| {:<15} | {:<35} | {:<6} | {:<6} | {:<6} | {:<43} | {:<25} |"
        sep_line = "-" * 155
        
        print(sep_line)
        print(row_format.format(*headers))
        print(sep_line)
        for row in results:
            print(row_format.format(*row))
        print(sep_line)

if __name__ == "__main__":
    run_stress_test()
