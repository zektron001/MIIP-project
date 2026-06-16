import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from huggingface_hub.utils import disable_progress_bars
from langchain_core.documents import Document

disable_progress_bars()
load_dotenv()

CHROMA_PATH = "data/chroma_db"
FAISS_PATH = "data/faiss_index"


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
        chunk_size=2000, chunk_overlap=400, add_start_index=True
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


def load_and_index_pdf_FAISS(pdf_path: str):
    """Load a PDF, split into chunks, embed and store in FAISS."""

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
        chunk_size=2000, chunk_overlap=400, add_start_index=True
    )
    chunks = splitter.split_documents(merged)

    # map each chunk back to the page it starts on
    for c in chunks:
        off = c.metadata["start_index"]
        c.metadata["page"] = max(p for s, p in page_starts if s <= off)

    print(f"Split into {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectorstore.save_local(FAISS_PATH)
    print(f"Stored {len(chunks)} chunks in FAISS")
    return vectorstore


def load_existing_faiss_index():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore


def ask_document(question, vectorstore, llm=None, reranker=None, return_sources=False):
    """Ask a question and get an answer from the document."""
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini")
    if reranker is None:
        reranker = CrossEncoder("BAAI/bge-reranker-base")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    docs = retriever.invoke(question)
    pairs = [(question, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    docs = [d for _, d in ranked[:3]]
    context = "\n".join(doc.page_content for doc in docs)

    pages = sorted({doc.metadata["page"] + 1 for doc in docs if "page" in doc.metadata})
    print(f"\n[Sources: pages {pages}]")

    messages = [
        {
            "role": "system",
            "content": "Answer questions using only the context provided. If the answer is not in the context, say so.",
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    response = llm.invoke(messages)

    if return_sources:
        return {"answer": response.content, "pages": pages}
    return response.content
