import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "data/chroma_db"


def load_and_index_pdf(pdf_path: str):
    """Load a PDF, split into chunks, embed and store in ChromaDB."""

    print(f"Loading PDF: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
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

    # Get relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)

    # Build context from chunks
    context = "\n\n".join(doc.page_content for doc in docs)

    # Show source pages
    pages = set(doc.metadata.get("page", 0) + 1 for doc in docs)
    print(f"\n[Sources: pages {sorted(pages)}]")

    # Ask GPT with context
    messages = [
        {
            "role": "system",
            "content": "Answer questions using only the context provided. If the answer is not in the context, say so.",
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    response = llm.invoke(messages)
    return response.content
