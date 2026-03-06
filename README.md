# rag_qa_project

A simple retrieval-augmented question answering system using embeddings and vector search.

## Goal
Build a mini RAG pipeline that:
- loads documents
- chunks them
- embeds them
- indexes them with FAISS
- retrieves relevant chunks for a query
- optionally generates an answer from retrieved evidence
