import os
from chat import ask
from rag import load_and_index_pdf, load_existing_index, ask_document

CHROMA_PATH = "chroma_db"
PDF_PATH = "data/document.pdf"


def main():
    while True:  # ← keeps showing menu until quit
        print("\nMIIP — Week 2")
        print("1. Chat normally")
        print("2. RAG Use ChromaDB + OpenAIEmbeddings + reranker")
        print("3. RAG Use FAISS + OpenAIEmbeddings + reranker")
        print("type quit to exit")
        choice = input("\nChoose: ").strip()

        if choice == "1":
            print("\nNormal chat mode. Type 'back' to go back. Type 'quit' to exit.\n")
            while True:
                question = input("You: ").strip()
                if question.lower() == "back":
                    break  # ← exits inner loop, outer loop shows menu again
                if question.lower() == "quit":
                    print("\nTake care!")
                    return  # ← exits everything
                if question:
                    print(f"\nGPT: {ask(question)}\n")

        elif choice == "2":
            if not os.path.exists(PDF_PATH):
                print(f"No PDF found at {PDF_PATH}")
                continue  # ← goes back to menu

            if not os.path.exists(CHROMA_PATH):
                vectorstore = load_and_index_pdf(PDF_PATH)
            else:
                print("Loading existing index...")
                vectorstore = load_existing_index()

            print("\nPDF mode. Type 'back' to go back. Type 'quit' to exit.\n")
            while True:
                question = input("You: ").strip()
                if question.lower() == "back":
                    break  # ← exits inner loop, outer loop shows menu again
                if question.lower() == "quit":
                    print("\nTake care!")
                    return  # ← exits everything
                if question:
                    answer = ask_document(question, vectorstore)
                    print(f"\nGPT: {answer}\n")
        elif choice == "3":
            if not os.path.exists(PDF_PATH):
                print(f"No PDF found at {PDF_PATH}")
                continue  # ← goes back to menu

            if not os.path.exists(CHROMA_PATH):
                vectorstore = load_and_index_pdf(PDF_PATH)
            else:
                print("Loading existing index...")
                vectorstore = load_existing_index()

            print("\nPDF mode. Type 'back' to go back. Type 'quit' to exit.\n")
            while True:
                question = input("You: ").strip()
                if question.lower() == "back":
                    break  # ← exits inner loop, outer loop shows menu again
                if question.lower() == "quit":
                    print("\nTake care!")
                    return  # ← exits everything
                if question:
                    answer = ask_document(question, vectorstore)
                    print(f"\nGPT: {answer}\n")
        elif choice.lower() == "quit":
            print("\nTake care!")
            break  # ← exits the outer while True


if __name__ == "__main__":
    main()
