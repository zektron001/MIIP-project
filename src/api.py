"""
FastAPI backend that wraps your existing pipeline.

Run with:
    uvicorn api:app --reload

Then open http://localhost:8000 in your browser.

This imports your existing chat.py and rag.py UNCHANGED except for two small
backward-compatible tweaks to rag.py (see the message in chat). The heavy models
(LLM client + reranker) and the vector indexes are loaded ONCE at startup, not on
every request.
"""

from fastapi.staticfiles import StaticFiles  # with your imports
from fastapi.responses import Response
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder

# your existing code, imported as-is
from chat import ask
from rag import (
    load_and_index_pdf,
    load_existing_index,
    load_and_index_pdf_FAISS,
    load_existing_faiss_index,
    ask_document,
)

load_dotenv()

# keep these in sync with rag.py
CHROMA_PATH = "data/chroma_db"
FAISS_PATH = "data/faiss_index"
PDF_PATH = "data/document.pdf"

# everything loaded once at startup lives here
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts (and cleans up on shutdown)."""
    print("Loading models (this happens once)...")

    # load the expensive stuff a single time
    state["llm"] = ChatOpenAI(model="gpt-4o-mini")
    state["reranker"] = CrossEncoder("BAAI/bge-reranker-base")

    # ChromaDB vector store: load if it exists, else build from the PDF
    if os.path.exists(CHROMA_PATH):
        print("Loading existing Chroma index...")
        state["chroma"] = load_existing_index()
    elif os.path.exists(PDF_PATH):
        print("Building Chroma index from PDF...")
        state["chroma"] = load_and_index_pdf(PDF_PATH)
    else:
        state["chroma"] = None

    # FAISS vector store: same logic
    if os.path.exists(FAISS_PATH):
        print("Loading existing FAISS index...")
        state["faiss"] = load_existing_faiss_index()
    elif os.path.exists(PDF_PATH):
        print("Building FAISS index from PDF...")
        state["faiss"] = load_and_index_pdf_FAISS(PDF_PATH)
    else:
        state["faiss"] = None

    print("Ready. Open http://localhost:8000")
    yield
    state.clear()


app = FastAPI(title="MIIP Chatbot", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")  # near your routes
# allow a separately-hosted frontend to call this. Tighten allow_origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- request / response shapes ----
class ChatRequest(BaseModel):
    message: str
    mode: str = "normal"  # "normal" | "chroma" | "faiss"


class ChatResponse(BaseModel):
    answer: str
    pages: list[int] = []  # source pages, only for RAG modes


# ---- the one endpoint the frontend calls ----
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is empty.")

    if req.mode == "normal":
        return ChatResponse(answer=ask(message))

    if req.mode in ("chroma", "faiss"):
        vectorstore = state.get(req.mode)
        if vectorstore is None:
            raise HTTPException(
                status_code=503,
                detail=f"No {req.mode} index available. Put a PDF at {PDF_PATH} and restart.",
            )
        result = ask_document(
            message,
            vectorstore,
            llm=state["llm"],
            reranker=state["reranker"],
            return_sources=True,
        )
        return ChatResponse(answer=result["answer"], pages=result["pages"])

    raise HTTPException(status_code=400, detail=f"Unknown mode: {req.mode}")


# ---- serve the frontend ----
@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)  # "no content", silences the request
