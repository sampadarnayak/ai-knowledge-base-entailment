## AI Knowledge Base With Entailment

An intelligent Question Answering system grounded in official DoPT (Department of Personnel and Training) Office Memoranda, built to respond accurately to natural language queries on Indian Central Government service rules spanning retirement, promotion, deputation, pay, disciplinary proceedings, and reservation.

Policy documents are processed through a pipeline of rule extraction, NLI-based knowledge hierarchy construction using DeBERTa-v3-base, immutable fact extraction, semantic retrieval using SentenceTransformers, and answer generation via LLaMA-3.3-70b through the Groq API — deployed as a Flask web application with a chat interface.

The system is built with a domain-agnostic architecture and can be adapted to any structured policy, legal, or regulatory corpus by replacing the source documents, updating the prompts in `app.py` and `run_qa.py`, and re-running the build scripts.
