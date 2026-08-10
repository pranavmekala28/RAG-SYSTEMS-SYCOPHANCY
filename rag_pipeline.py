"""
rag_pipeline.py  —  RAG experiment harness (3 knobs + generator)
================================================================
Your study's question: do the measured benefits of retrieval choices
(chunk size, reranking, hybrid search) hold up — or flip — depending on
which evaluator grades them? This file builds the CONFIGURABLE pipeline
whose outputs we'll later score with multiple judges (RAGAS / DeepEval).

DESIGN RULE (this is what makes it publishable, not a tuning exercise):
the CONFIG block below is the ONLY place you turn knobs. Change ONE
variable at a time so every result is interpretable.

Run:   python rag_pipeline.py
First run downloads the embedding model (~80MB) and, if reranking is on,
the cross-encoder (~80MB). Both cache after that.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


# ============================================================
# 0. CONFIG  —  the ONLY place you turn knobs. Sweep ONE at a time.
# ============================================================
@dataclass
class Config:
    chunk_size: int = 60          # KNOB 1: words per chunk
    chunk_overlap: int = 15       # words shared between neighbouring chunks
    retrieval_mode: str = "bm25" # KNOB 2: "dense" | "bm25" | "hybrid"
    use_reranker: bool = False    # KNOB 3: cross-encoder rerank on/off
    top_k: int = 4                # how many chunks to feed the generator
    hybrid_alpha: float = 0.5     # hybrid only: 1.0 = all dense, 0.0 = all bm25

CONFIG = Config()


# ============================================================
# 1. TOY CORPUS  —  longer passages so chunk size actually matters.
#    Swap this for a real benchmark (HotpotQA / RAGBench) later.
# ============================================================
DOCUMENTS = [
    # 0 — the passage that truly answers the query (spread across sentences)
    "Retrieval augmented generation, or RAG, grounds a language model in "
    "external documents that are fetched at query time. Instead of relying "
    "only on parameters learned during training, the model is given passages "
    "retrieved from a corpus and asked to answer using them. Because the "
    "answer must be supported by the supplied passages, the model is far less "
    "likely to fabricate facts. This grounding is the main reason RAG reduces "
    "hallucination compared with a bare language model that answers from memory "
    "alone, and it also lets the system cite the source of each claim.",

    # 1 — embeddings (related, lexical overlap on 'vector', 'meaning')
    "An embedding model maps a piece of text to a dense vector of numbers so "
    "that passages with similar meaning land close together in vector space. "
    "Sentence-transformers such as all-MiniLM produce 384-dimensional vectors. "
    "Similarity between a query vector and a document vector is usually measured "
    "with cosine similarity, which is what dense retrieval ranks on.",

    # 2 — chunking (related to a knob)
    "Long documents are split into smaller chunks before indexing. Chunk size "
    "is a tradeoff: small chunks give precise matches but may cut a fact in "
    "half, while large chunks keep context together but dilute the relevant "
    "sentence with surrounding text. Overlapping chunks reduce the risk of "
    "splitting an answer across a boundary.",

    # 3 — reranking (related to a knob)
    "A reranker is a cross-encoder that reads the query and a candidate passage "
    "together and outputs a relevance score. It is slower than a bi-encoder, so "
    "it is applied only to a shortlist of candidates that a fast retriever "
    "returned first. Rerankers often improve precision at the top of the list.",

    # 4 — bm25 / lexical (related to a knob, strong keyword overlap)
    "BM25 is a classic lexical retrieval function that scores documents by "
    "exact keyword overlap with the query, weighting rare terms more heavily. "
    "It ignores meaning, so it fails on paraphrases, but it is unbeatable when "
    "the query and the answer share the same words. Hybrid search combines BM25 "
    "scores with dense similarity to get the best of both.",

    # 5 — distractor (unrelated topic, should NOT be retrieved)
    "Photosynthesis is the process by which green plants convert sunlight, "
    "water, and carbon dioxide into glucose and oxygen. It takes place in the "
    "chloroplasts, where the pigment chlorophyll absorbs light energy. The "
    "process is the foundation of most food chains on Earth.",
]

QUERY = "How does retrieval augmented generation reduce hallucination?"


# ============================================================
# 2. CHUNKING  (KNOB 1 lives here)
# ============================================================
def chunk_documents(docs, size, overlap):
    """Split each doc into word windows of `size`, sliding by (size - overlap)."""
    chunks, source_doc = [], []
    step = max(1, size - overlap)
    for doc_id, doc in enumerate(docs):
        words = doc.split()
        for start in range(0, len(words), step):
            piece = words[start:start + size]
            if not piece:
                continue
            chunks.append(" ".join(piece))
            source_doc.append(doc_id)
            if start + size >= len(words):
                break  # last window reached this doc's end
    return chunks, source_doc


# ============================================================
# 3. SCORING  (KNOB 2 lives here)
# ============================================================
def minmax(x):
    """Scale an array to [0, 1] so dense and bm25 scores are comparable."""
    x = np.asarray(x, dtype="float32")
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)


def score_chunks(query, chunks, chunk_emb, bm25, embedder, cfg):
    """Return one score per chunk according to the retrieval mode."""
    q = embedder.encode([query], normalize_embeddings=True).astype("float32")[0]
    dense = chunk_emb @ q                                   # cosine (vectors are normalized)
    lexical = np.asarray(bm25.get_scores(query.lower().split()), dtype="float32")

    if cfg.retrieval_mode == "dense":
        return dense
    if cfg.retrieval_mode == "bm25":
        return lexical
    if cfg.retrieval_mode == "hybrid":
        return cfg.hybrid_alpha * minmax(dense) + (1 - cfg.hybrid_alpha) * minmax(lexical)
    raise ValueError(f"unknown retrieval_mode: {cfg.retrieval_mode!r}")


# ============================================================
# 4. RERANKING  (KNOB 3 lives here)
# ============================================================
_RERANKER = None  # loaded lazily, only if reranking is switched on

def get_reranker():
    global _RERANKER
    if _RERANKER is None:
        print("  loading cross-encoder reranker (first time downloads ~80MB)...")
        _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _RERANKER


def select_topk(query, ranked_idxs, chunks, cfg):
    """Take the candidate pool and narrow to top_k, reranking if enabled."""
    if not cfg.use_reranker:
        return ranked_idxs[:cfg.top_k]
    reranker = get_reranker()
    pairs = [(query, chunks[i]) for i in ranked_idxs]
    rr_scores = reranker.predict(pairs)
    order = np.argsort(rr_scores)[::-1]
    return [ranked_idxs[i] for i in order[:cfg.top_k]]


# ============================================================
# 5. GENERATOR  (the LLM that answers from retrieved context)
# ============================================================
def generate_answer(query, context_chunks):
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    # ---- PLUG IN YOUR LLM HERE (next build step) ----
    # Anthropic example — set ANTHROPIC_API_KEY in your environment first:
    #   import anthropic
    #   client = anthropic.Anthropic()
    #   msg = client.messages.create(
    #       model="claude-sonnet-4-5", max_tokens=300,
    #       messages=[{"role": "user", "content": prompt}])
    #   return msg.content[0].text
    #
    # Until an LLM is wired, return the assembled context so the pipeline
    # runs end to end and you can SEE what the generator would receive:
    return "[no LLM wired yet — this is the context the generator would get]\n" + context


# ============================================================
# 6. RUN ONE CONFIG END TO END
# ============================================================
def main():
    cfg = CONFIG
    print(f"\n=== CONFIG ===\n{cfg}\n")

    chunks, source_doc = chunk_documents(DOCUMENTS, cfg.chunk_size, cfg.chunk_overlap)
    print(f"{len(DOCUMENTS)} docs -> {len(chunks)} chunks "
          f"(size={cfg.chunk_size}, overlap={cfg.chunk_overlap})\n")

    print("Embedding chunks...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_emb = embedder.encode(chunks, normalize_embeddings=True).astype("float32")
    bm25 = BM25Okapi([c.lower().split() for c in chunks])

    scores = score_chunks(QUERY, chunks, chunk_emb, bm25, embedder, cfg)

    # candidate pool: grab extra if we're going to rerank, else just top_k
    pool_size = cfg.top_k * 4 if cfg.use_reranker else cfg.top_k
    pool = list(np.argsort(scores)[::-1][:pool_size])
    final = select_topk(QUERY, pool, chunks, cfg)

    print(f"=== RETRIEVED TOP-{cfg.top_k} CHUNKS ===")
    for rank, i in enumerate(final, 1):
        preview = chunks[i][:110].replace("\n", " ")
        print(f"{rank}. (from doc {source_doc[i]})  {preview}...")

    print("\n=== GENERATED ANSWER ===")
    print(generate_answer(QUERY, [chunks[i] for i in final]))


if __name__ == "__main__":
    main()