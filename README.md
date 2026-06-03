# MIIP-project
AI engineering portfolio project — under development

Building a production-grade multimodal incident triage system
using LangGraph agents, RAG, HuggingFace models, and a full
LLM eval harness.

- [x] Week 1: OpenAI API + project scaffold
- [ ] Week 2: RAG pipeline with PDF + ChromaDB
- [ ] Week 3: FastAPI + deployed live URL
- [ ] Week 4–7: Multi-agent system with LangGraph
- [ ] Week 8–12: Full MIIP multimodal pipeline + eval

Python · OpenAI · LangChain · LangGraph · pgvector
HuggingFace · FastAPI · RAGAS · LangSmith · Docker

```
pip install -r requirements.txt
cp .env.example .env  # add your OPENAI_API_KEY
python src/main.py
```