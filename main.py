"""
Mini Retrieval-Augmented QA System

Pipeline:
- Load document corpus
- Split documents into chunks
- Embed chunks using SentenceTransformers
- Store embeddings in a FAISS index
- Retrieve top-k relevant chunks for a query
- Generate an answer using FLAN-T5

This project demonstrates a minimal end-to-end RAG pipeline.
"""

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np
import torch


# Load document corpus
with open("data/documents.txt", "r") as f:
    raw_docs = [line.strip() for line in f if line.strip()]

print("Loaded documents:", len(raw_docs))


# Chunk documents into smaller pieces for retrieval
def chunk_text(text, chunk_size=12):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

    return chunks


chunks = []
for doc in raw_docs:
    chunks.extend(chunk_text(doc))

print("Created chunks:", len(chunks))


# Sentence embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_embeddings = embed_model.encode(chunks)
chunk_embeddings = np.array(chunk_embeddings).astype("float32")
faiss.normalize_L2(chunk_embeddings)


# Build FAISS vector index
dim = chunk_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(chunk_embeddings)

print("FAISS index size:", index.ntotal)


# Load generation model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Generation device:", device)

gen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)


# Retrieval function
def retrieve_chunks(query, k=3):
    q_emb = embed_model.encode([query])
    q_emb = np.array(q_emb).astype("float32")
    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, k)
    retrieved = [chunks[i] for i in indices[0]]

    return retrieved, scores, indices


# Example queries
queries = [
    "What is FAISS used for?",
    "Where is Mount Everest?",
    "What is Python used for?",
    "Which river flows into the Mediterranean?"
]


# Retrieval and generation
for query in queries:
    print("\n" + "=" * 70)
    print("Question:", query)

    retrieved, scores, indices = retrieve_chunks(query, k=3)

    print("\nRetrieved chunks:")
    for rank, i in enumerate(indices[0]):
        print(f"{rank + 1}. {chunks[i]} | score = {round(float(scores[0][rank]), 4)}")

    context = " ".join(retrieved)

    prompt = f"""Answer the question using only the context below.

Context:
{context}

Question:
{query}

Answer:"""

    inputs = gen_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    outputs = gen_model.generate(
        **inputs,
        max_new_tokens=40
    )

    answer = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nGenerated answer:")
    print(answer)


print("\nDone.")
print("Note: Retrieval quality depends on chunking strategy, embedding quality, and ranking accuracy.")
