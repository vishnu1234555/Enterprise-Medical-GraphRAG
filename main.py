from src.agent.rag_bot import MedicalGraphRAG


def main():
    bot = MedicalGraphRAG()
    while True:
        user_question = input("\nAsk a medical graph question (or 'exit'): ").strip()
        if not user_question:
            continue
        if user_question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        disease_id, _ = bot.semantic_vector_search(user_question)
        if not disease_id:
            print("No relevant disease found in vector index.")
            continue

        context = bot.deterministic_traversal(disease_id)
        draft = bot.generate_draft_response(user_question, context)
        metrics = bot.evaluate_rag_triad(user_question, draft, context)

        if float(metrics["faithfulness"]) < 1.0:
            print("[CRITICAL WARNING] Hallucination detected by critic. Draft discarded.")
            print(context)
            continue

        print(draft)


if __name__ == "__main__":
    main()
