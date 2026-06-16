# MIIP-project
***AI engineering portfolio project — [under development]***

Building a production-grade multimodal incident triage system
using LangGraph agents, RAG, HuggingFace models, and a full
LLM eval harness.

***Implementing different types of RAG Strategy***

1. Normal chat API (OpenAI)
2. RAG Use ChromaDB + OpenAIEmbeddings + reranker
3. RAG Use FAISS + OpenAIEmbeddings + reranker"

**Features**


**Progress Timeline**
- [x] Week 1: OpenAI API + project scaffold
- [x] Week 2: RAG pipeline with PDF + ChromaDB
- [x] week 3: Implement Reranking & different similarity check
- [ ] Week 4: FastAPI + deployed live URL(Fast Api - Done)  
- [ ] Week 5–7: Multi-agent system with LangGraph
- [ ] Week 8–12: Full MIIP multimodal pipeline + eval

Python · OpenAI · LangChain · LangGraph · pgvector
HuggingFace · FastAPI · RAGAS · LangSmith · Docker

```
pip install -r requirements.txt
cp .env.example .env  # add your OPENAI_API_KEY
.venv\Scripts\activate.bat # activate your virtual enviroment
python src/main.py #To run and execute the program 
uvicorn api:app --reload --app-dir src #Run the UI
```