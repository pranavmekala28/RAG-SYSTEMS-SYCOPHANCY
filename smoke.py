from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# tiny demo corpus — a few "documents" to search over
docs = [
    "The Eiffel Tower is located in Paris, France.",
    "Python is a popular programming language for data science.",
    "RAG combines a retriever with a language model to answer questions.",
    "Mount Everest is the tallest mountain on Earth.",
    "Vector databases store embeddings for fast similarity search.",
    "Photosynthesis lets plants convert sunlight into energy.",
]

# 1. load the embedding model (same one we just tested)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. embed every doc and build a FAISS index
doc_embeddings = model.encode(docs, normalize_embeddings=True)
index = faiss.IndexFlatIP(doc_embeddings.shape[1])   # cosine similarity
index.add(np.asarray(doc_embeddings, dtype="float32"))

# 3. ask a question
query = "How do RAG systems work?"
q_emb = model.encode([query], normalize_embeddings=True)
scores, ids = index.search(np.asarray(q_emb, dtype="float32"), k=3)

# 4. show the top matches
print(f"\nQuery: {query}\n")
print("Top 3 matches:")
for rank, (i, s) in enumerate(zip(ids[0], scores[0]), 1):
    print(f"{rank}. (score {s:.3f}) {docs[i]}")