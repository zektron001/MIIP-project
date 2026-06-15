import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from huggingface_hub.utils import disable_progress_bars
from langchain_core.documents import Document

disable_progress_bars()
load_dotenv()

CHROMA_PATH = "data/chroma_db"


def load_and_index_pdf(pdf_path: str):
    """Load a PDF, split into chunks, embed and store in ChromaDB."""

    print("Loading PDF")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    full_text = ""
    page_starts = []  # (char_offset, page_number)
    for d in documents:
        page_starts.append((len(full_text), d.metadata.get("page", 0)))
        full_text += d.page_content + "\n"

    merged = [Document(page_content=full_text, metadata={"source": pdf_path})]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    chunks = splitter.split_documents(merged)

    # map each chunk back to the page it starts on
    for c in chunks:
        off = c.metadata["start_index"]
        c.metadata["page"] = max(p for s, p in page_starts if s <= off)

    print(f"Split into {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB")
    return vectorstore


def load_existing_index():
    """Load an already-indexed ChromaDB (so you don't re-index every time)."""
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return vectorstore


def ask_document(question: str, vectorstore) -> str:
    """Ask a question and get an answer from the document."""
    llm = ChatOpenAI(model="gpt-4o-mini")
    reranker = CrossEncoder("Qwen/Qwen3-Reranker-0.6B")
    # Get relevant chunks
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 20}
    )  # 1. retrieve 20 chunks
    docs = retriever.invoke(question)
    pairs = [(question, d.page_content) for d in docs]  # 3. pair up
    scores = reranker.predict(pairs)  # 4. RERANK → score the 20
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    docs = [d for _, d in ranked[:3]]
    # Build context from chunks
    context = "\n".join(doc.page_content for doc in docs)

    # Show source pages
    pages = sorted({doc.metadata["page"] + 1 for doc in docs if "page" in doc.metadata})
    print(f"\n[Sources: pages {pages}]")

    # Ask GPT with context - Augmented Generation (RAG)
    messages = [
        {
            "role": "system",
            "content": "Answer questions using only the context provided. If the answer is not in the context, say so.",
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    # generate answer
    response = llm.invoke(messages)
    return response.content
