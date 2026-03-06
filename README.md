
  # Mini RAG QA System

A minimal **retrieval-augmented question answering system** built with sentence embeddings, FAISS vector search, and FLAN-T5 generation.

## Overview

Large language models often answer questions using only their internal knowledge.  
Retrieval-augmented generation (RAG) improves this by retrieving relevant external information and grounding the answer in that evidence.

This project implements a small end-to-end RAG pipeline that:

1. loads a collection of documents  
2. splits documents into smaller chunks  
3. embeds chunks using a sentence embedding model  
4. stores embeddings in a FAISS vector index  
5. retrieves the most relevant chunks for a query  
6. generates an answer using FLAN-T5 conditioned on the retrieved context  

The goal of the project was to understand how **retrieval and generation interact in modern NLP systems**.

---

# Pipeline

### Document Loading

The system loads a small text corpus from `documents.txt`.

---

### Chunking

Documents are split into smaller chunks so retrieval can focus on relevant pieces of information rather than entire documents.

Chunking improves retrieval specificity but introduces trade-offs when useful context spans multiple chunks.

---

### Embedding

Each chunk is embedded using:

`sentence-transformers/all-MiniLM-L6-v2`

This model produces dense vectors that place semantically similar text near each other in embedding space.

---

### Vector Search

Embeddings are **L2 normalized** and stored in a FAISS index using:

`faiss.IndexFlatIP`

This approximates **cosine similarity search** between query and document vectors.

---

### Retrieval

Given a query:

1. The query is embedded
2. The FAISS index retrieves the **top-k most similar chunks**
3. Retrieved chunks are combined into a context block

---

### Generation

The retrieved context is passed to:

`google/flan-t5-small`

The model generates an answer **conditioned on the retrieved evidence**, ensuring responses are grounded in the document corpus.

---

# Example Results

### Query

**What is FAISS used for?**

Retrieved chunks:

1. FAISS is a library for efficient similarity search over dense vectors  
2. It is often used in retrieval-augmented generation systems  

Generated answer:

`similarity search`

---

### Query

**Where is Mount Everest?**

Generated answer:

`Himalayas`

---

### Query

**What is Python used for?**

Generated answer:

`machine learning, web development, and automation`

---

### Query

**Which river flows into the Mediterranean?**

Generated answer:

`The Nile`

---

# Technical Insights

### Retrieval quality depends strongly on chunking

Smaller chunks improve retrieval precision but can separate useful context across multiple segments.

### Embedding retrieval is imperfect

Even when the correct information exists in the corpus, the most relevant chunk is not always ranked first.

### Top-k retrieval improves robustness

Retrieving multiple chunks helps mitigate ranking errors and increases the likelihood that the correct evidence appears in context.

### Generation quality depends on retrieval quality

The answer generator can only produce accurate responses if relevant information is retrieved first.

---

# Technologies Used

- Python
- SentenceTransformers
- FAISS
- NumPy
- PyTorch
- HuggingFace Transformers
- FLAN-T5-small
- Google Colab GPU

---

# Files

```
main.py
```

End-to-end RAG pipeline including chunking, embedding, retrieval, and generation.

```
documents.txt
```

Example document corpus.

```
requirements.txt
```

Python dependencies.

---

# Future Work

Possible extensions include:

- larger document collections  
- improved chunking strategies  
- saving and loading FAISS indexes  
- evaluation metrics for retrieval quality  
- reranking methods for improved retrieval accuracy  
- interactive QA interfaces
