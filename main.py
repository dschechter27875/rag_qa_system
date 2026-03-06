from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np
import torch

# -------------------------
# 1 Load documents
# -------------------------

with open("data/documents.txt", "r") as f:
    raw_docs = [line.strip() for line in f if line.strip()]

print("Loaded documents:", len(raw_docs))

# -------------------------
# 2 Chunk documents
# -------------------------

def chunk_text(text, chunk_size=12):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

chunks = []
for doc in raw_docs:
    chunks.extend(chunk_text(doc))

print("Created chunks:", len(chunks))

# -------------------------
# 3 Embedding model
# -------------------------

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_embeddings = embed_model.encode(chunks)
chunk_embeddings = np.array(chunk_embeddings).astype("float32")
faiss.normalize_L2(chunk_embeddings)

# -------------------------
# 4 Build FAISS index
# -------------------------

dim = chunk_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(chunk_embeddings)

print("FAISS index size:", index.ntotal)

# -------------------------
# 5 Load FLAN-T5
# -------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Generation device:", device)

gen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)

# -------------------------
# 6 Example questions
# -------------------------

queries = [
    "What is FAISS used for?",
    "Where is Mount Everest?",
    "What is Python used for?",
    "Which river flows into the Mediterranean?"
]

# -------------------------
# 7 Retrieval + generation
# -------------------------

for query in queries:
    print("\n" + "="*70)
    print("Question:", query)

    # Retrieve
    q_emb = embed_model.encode([query])
    q_emb = np.array(q_emb).astype("float32")
    faiss.normalize_L2(q_emb)

    k = 3
    scores, indices = index.search(q_emb, k)
    retrieved = [chunks[i] for i in indices[0]]

    print("\nRetrieved chunks:")
    for r in retrieved:
        print("-", r)

    # Build context
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
