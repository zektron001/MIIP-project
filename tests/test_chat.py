import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from rag import load_existing_index, ask_document


def test_rag_returns_answer():
    """RAG pipeline should return a non-empty answer."""
    if not os.path.exists("data/chroma_db"):
        print("Skipping — no index found. Run main.py option 2 first.")
        return

    vectorstore = load_existing_index()
    answer = ask_document("Summarise this document in one sentence.", vectorstore)
    assert isinstance(answer, str)
    assert len(answer) > 10
    print(f"Answer: {answer}")


if __name__ == "__main__":
    test_rag_returns_answer()
