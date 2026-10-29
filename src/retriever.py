"""
Hybrid retrieval over schemes_data.json.

Always available:
  * BM25 (rank_bm25, whitespace + Telugu tokens)
  * Character TF-IDF cosine (sklearn) — a lexical vector retriever, not a
    neural embedding model.

Optional:
  * Neural dense retrieval if sentence-transformers can load a multilingual
    model. Disabled by default so the dashboard does not download weights
    on first launch. Set TELUGU_CIVIC_DENSE=1 to attempt it.

Hybrid rank: Reciprocal Rank Fusion of BM25 + char-TF-IDF (+ neural if loaded),
then a small bonus when the chunk category matches the predicted intent.
"""
import os
import re
import hashlib
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from kb import load_chunks
from paths import CHROMA_DIR

TELUGU_OR_WORD = re.compile(r"[\u0c00-\u0c7f]+|[A-Za-z0-9]+")
def tokenize(text: str):
    return TELUGU_OR_WORD.findall(text or "")


def _with_suffix_variants(tokens):
    variants = list(tokens)
    for token in tokens:
        match = re.match(r"^(.{3,})(?:కి|కు|లో|తో|ను|ని|లు|ల)$", token)
        if match:
            variants.append(match.group(1))
    return variants


def _index_text(chunk: dict) -> str:
    return f"{chunk['scheme']} {chunk['category']} {chunk['content']}"


class HybridRetriever:
    def __init__(self, chunks=None, enable_neural=None, bm25_weight=1.8,
                 dense_weight=1.0, rrf_k=60, suffix_stripping=True,
                 persist_directory=None):
        self.chunks = list(chunks if chunks is not None else load_chunks())
        self.bm25_weight = float(bm25_weight)
        self.dense_weight = float(dense_weight)
        self.rrf_k = int(rrf_k)
        self.suffix_stripping = bool(suffix_stripping)
        self.persist_directory = persist_directory or CHROMA_DIR
        self.corpus = [_index_text(c) for c in self.chunks]
        tokenized = [
            _with_suffix_variants(tokenize(t)) if self.suffix_stripping else tokenize(t)
            for t in self.corpus
        ]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

        self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self.tfidf_matrix = self.tfidf.fit_transform(self.corpus) if self.corpus else None

        if enable_neural is None:
            enable_neural = os.environ.get("TELUGU_CIVIC_DENSE", "").strip() in {"1", "true", "yes"}
        self.neural_model = None
        self.neural_embeddings = None
        self.neural_error = None
        if enable_neural:
            self._try_load_neural()

    def _try_load_neural(self):
        try:
            from sentence_transformers import SentenceTransformer

            name = os.environ.get(
                "TELUGU_CIVIC_DENSE_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            )
            self.neural_model = SentenceTransformer(name)
            cache_path = os.path.join(self.persist_directory, "dense_embeddings.npz")
            cache_key = hashlib.sha256(
                (name + "\n" + "\n".join(self.corpus)).encode("utf-8")
            ).hexdigest()
            cached = None
            if os.path.exists(cache_path):
                cached = np.load(cache_path, allow_pickle=False)
            if cached is not None and str(cached["key"]) == cache_key:
                self.neural_embeddings = cached["embeddings"]
            else:
                self.neural_embeddings = self.neural_model.encode(
                    self.corpus, convert_to_numpy=True, show_progress_bar=False
                )
                os.makedirs(self.persist_directory, exist_ok=True)
                np.savez_compressed(cache_path, key=cache_key, embeddings=self.neural_embeddings)
            self._sync_chroma()
        except Exception as exc:  # model missing, no network, etc.
            self.neural_model = None
            self.neural_embeddings = None
            self.neural_error = str(exc)

    def _sync_chroma(self):
        """Best-effort persistent mirror; lexical retrieval remains usable without Chroma."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_directory)
            collection = client.get_or_create_collection("telugu_civic_chunks")
            ids = [str(c["doc_id"]) for c in self.chunks]
            collection.upsert(
                ids=ids,
                embeddings=self.neural_embeddings.tolist(),
                documents=self.corpus,
                metadatas=[{k: str(c.get(k, "")) for k in (
                    "scheme", "category", "source_url", "state", "authority",
                    "document_id"
                )} for c in self.chunks],
            )
            self.chroma_collection = collection
        except Exception as exc:
            self.chroma_collection = None
            self.chroma_error = str(exc)

    def status(self):
        neural = "operational" if self.neural_model is not None else "not_loaded"
        return {
            "bm25": "operational",
            "char_tfidf": "operational",
            "neural_dense": neural,
            "neural_error": self.neural_error,
            "hybrid": "operational",
            "n_chunks": len(self.chunks),
            "weights": {"bm25": self.bm25_weight, "dense": self.dense_weight},
            "chroma": "operational" if getattr(self, "chroma_collection", None) else "not_loaded",
        }

    def _bm25_scores(self, query: str) -> np.ndarray:
        if not self.bm25:
            return np.zeros(len(self.chunks))
        toks = tokenize(query)
        if self.suffix_stripping:
            toks = _with_suffix_variants(toks)
        if not toks:
            return np.zeros(len(self.chunks))
        return np.asarray(self.bm25.get_scores(toks), dtype=float)

    def _tfidf_scores(self, query: str) -> np.ndarray:
        if self.tfidf_matrix is None:
            return np.zeros(len(self.chunks))
        q = self.tfidf.transform([query or ""])
        return cosine_similarity(q, self.tfidf_matrix)[0]

    def _neural_scores(self, query: str) -> Optional[np.ndarray]:
        if self.neural_model is None or self.neural_embeddings is None:
            return None
        q = self.neural_model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        if getattr(self, "chroma_collection", None) is not None:
            try:
                response = self.chroma_collection.query(
                    query_embeddings=q.tolist(), n_results=len(self.chunks),
                    include=["distances"]
                )
                scores = np.zeros(len(self.chunks), dtype=float)
                by_id = {str(c["doc_id"]): i for i, c in enumerate(self.chunks)}
                for doc_id, distance in zip(response["ids"][0], response["distances"][0]):
                    if doc_id in by_id:
                        scores[by_id[doc_id]] = 1.0 - float(distance)
                return scores
            except Exception:
                pass
        denom = (
            np.linalg.norm(q, axis=1, keepdims=True)
            * np.linalg.norm(self.neural_embeddings, axis=1, keepdims=True).T
        )
        denom = np.maximum(denom, 1e-9)
        return (q @ self.neural_embeddings.T / denom)[0]

    @staticmethod
    def _rrf(score_lists, k=60, weights=None) -> np.ndarray:
        n = len(score_lists[0])
        fused = np.zeros(n)
        for pos, scores in enumerate(score_lists):
            order = np.argsort(-scores)
            ranks = np.empty(n, dtype=int)
            ranks[order] = np.arange(1, n + 1)
            weight = weights[pos] if weights else 1.0
            fused += weight / (k + ranks)
        return fused

    def search(self, query: str, intent=None, top_k=5, in_scope=True,               scheme=None, category=None, category_uncertain=False):
        if not self.chunks:
            return []
        if not (query or "").strip():
            return []
        bm25 = self._bm25_scores(query)
        tfidf = self._tfidf_scores(query)
        parts = [bm25, tfidf]
        neural = self._neural_scores(query)
        if neural is not None:
            parts.append(neural)
        weights = [self.bm25_weight, 1.0]
        if neural is not None:
            weights.append(self.dense_weight)
        fused = self._rrf(parts, k=self.rrf_k, weights=weights)
        eligible = np.ones(len(self.chunks), dtype=bool)
        if scheme:
            eligible &= np.array([c.get("scheme") == scheme for c in self.chunks])
        if category:
            # A chunk qualifies only if its primary category matches the
            # requested category. Multi-topic documents that list the category
            # in supported_categories are NOT included here: the answer layer
            # owns the hard filter, and the retriever must return passages that
            # are actually about the requested topic (e.g., eligibility),
            # not documents that merely mention it in a supported_categories
            # field. This prevents application-procedure documents from
            # surfacing for eligibility questions.
            primary = np.array([c.get("category") == category for c in self.chunks])
            eligible &= primary
            # Category as a rank hint: a primary-category chunk gets a small
            # RRF bonus *only when the query category is reliable* (not
            # low-confidence classifier output). This reorders ranks so the
            # best-matching category appears first, without broadening the
            # result set.
        if category and not category_uncertain:
            for i, chunk in enumerate(self.chunks):
                if eligible[i] and chunk.get("category") == category:
                    fused[i] += 0.05
        if intent and in_scope:
            for i, chunk in enumerate(self.chunks):
                if eligible[i] and chunk.get("category") == intent:
                    fused[i] += 0.02
        fused[~eligible] = -np.inf

        order = np.argsort(-fused)
        results = []
        for idx in order:
            if not np.isfinite(fused[idx]):
                continue
            if len(results) >= top_k:
                break
            chunk = self.chunks[int(idx)]
            results.append({
                "doc_id": chunk["doc_id"],
                "scheme": chunk["scheme"],
                "category": chunk["category"],
                "source_url": chunk.get("source_url") or "",
                "content": chunk["content"],
                "score_hybrid": float(fused[idx]),
                "score_bm25": float(bm25[idx]),
                "score_tfidf": float(tfidf[idx]),
                "score_dense": float(neural[idx]) if neural is not None else None,
                "source_title": chunk.get("source_title", ""),
                "state": chunk.get("state", ""),
                "authority": chunk.get("authority", ""),
                "scheme_identifier": chunk.get("scheme_identifier", ""),
                "published_date": chunk.get("published_date"),
                "updated_date": chunk.get("updated_date"),
                "source_role": chunk.get("source_role", ""),
                "source_classification": chunk.get("source_classification", ""),
                "verification_status": chunk.get("verification_status", ""),
                # Temporal provenance must travel with the passage, otherwise
                # historical orders cannot be flagged as historical downstream.
                "temporal_scope": chunk.get("temporal_scope", ""),
                "supported_categories": list(chunk.get("supported_categories") or []),
                "issuing_authority": chunk.get("issuing_authority", ""),
                "fetched_at": chunk.get("fetched_at"),
                "rank": len(results) + 1,
            })
        return results
