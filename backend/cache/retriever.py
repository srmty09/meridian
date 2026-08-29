import os
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

_token_re = re.compile(r"[a-z0-9]+")


def _tokenize(text: str):
    return _token_re.findall(text.lower())


def _topk_desc(scores, k):
    k = min(k, len(scores))
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]


class Retriever:
    def __init__(self):
        self.entries = []
        self._embedder = None
        self._emb = None
        self._tfidf = None
        self._tfidf_matrix = None
        self._bm25 = None

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = SentenceTransformer(EMBED_MODEL)
        return self._embedder

    def __len__(self):
        return len(self.entries)

    def add(self, prompt: str, response: str, model: str, tokens: int):
        self.entries.append({"prompt": prompt, "response": response, "model": model, "tokens": tokens})
        self._reindex()

    def _reindex(self):
        prompts = [e["prompt"] for e in self.entries]
        self._emb = self.embedder.encode(prompts, normalize_embeddings=True).astype("float32")
        self._tfidf = TfidfVectorizer()
        self._tfidf_matrix = self._tfidf.fit_transform(prompts)
        self._bm25 = BM25Okapi([_tokenize(p) for p in prompts])

    def retrieve(self, prompt: str, top_k: int = TOP_K):
        if not self.entries:
            return []

        scores = {}

        def note(i, name, value):
            scores.setdefault(i, {}).update({name: float(value)})

        q = self.embedder.encode([prompt], normalize_embeddings=True).astype("float32")[0]
        dense_sims = self._emb @ q
        for i in _topk_desc(dense_sims, top_k):
            note(int(i), "dense", dense_sims[i])

        q_vec = self._tfidf.transform([prompt])
        tfidf_sims = cosine_similarity(q_vec, self._tfidf_matrix)[0]
        for i in _topk_desc(tfidf_sims, top_k):
            note(int(i), "tfidf", tfidf_sims[i])

        bm25_scores = np.asarray(self._bm25.get_scores(_tokenize(prompt)))
        for i in _topk_desc(bm25_scores, top_k):
            note(int(i), "bm25", bm25_scores[i])

        candidates = []
        for i, sc in scores.items():
            entry = dict(self.entries[i])
            entry["scores"] = {"tfidf": sc.get("tfidf", 0.0), "bm25": sc.get("bm25", 0.0), "dense": sc.get("dense", 0.0)}
            candidates.append(entry)

        candidates.sort(key=lambda e: e["scores"]["dense"], reverse=True)
        return candidates[:top_k]
