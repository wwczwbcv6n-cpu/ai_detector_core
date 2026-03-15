"""
rag_store.py — RAG (Retrieval-Augmented Generation) vector store for AI detection.

Stores 768-dim fusion embeddings alongside detection verdicts.
On new image, retrieves k nearest past cases and optionally augments
the model's probability with a distance-weighted vote.

Backend: FAISS (if installed) → fast ANN search
         numpy  (fallback)    → brute-force L2, adequate up to ~10k entries
Metadata: SQLite (persistent)
"""

import os
import json
import time
import sqlite3
import numpy as np
from pathlib import Path

try:
    import faiss
    _FAISS = True
except ImportError:
    _FAISS = False

EMBEDDING_DIM = 768
_DEFAULT_STORE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'rag_store'
)


class ImageRAGStore:
    """
    Vector store for image fusion embeddings + retrieval-augmented detection.

    Usage
    -----
    store = ImageRAGStore()

    # During analysis (rag=True):
    neighbors = store.retrieve(embedding, k=5)
    aug_prob  = store.augment_probability(model_prob, neighbors)

    # After analysis (auto-index the result):
    store.add(embedding, verdict="AI-Generated", confidence=0.92)
    """

    def __init__(
        self,
        store_dir: str = _DEFAULT_STORE_DIR,
        embedding_dim: int = EMBEDDING_DIM,
    ):
        self.store_dir     = Path(store_dir)
        self.embedding_dim = embedding_dim
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self._db_path    = self.store_dir / "rag_metadata.db"
        self._idx_path   = self.store_dir / "faiss.index"
        self._map_path   = self.store_dir / "id_map.json"

        # Ordered list: position i → SQLite row_id for FAISS vector i
        self._id_map: list = []

        self._init_db()
        self._init_index()

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _init_db(self):
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                verdict    TEXT    NOT NULL,
                confidence REAL    NOT NULL,
                timestamp  REAL    NOT NULL,
                metadata   TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _init_index(self):
        """Load persisted FAISS/numpy index or start fresh."""
        if _FAISS:
            if self._idx_path.exists() and self._map_path.exists():
                try:
                    self._index = faiss.read_index(str(self._idx_path))
                    with open(self._map_path, encoding='utf-8') as f:
                        self._id_map = json.load(f)
                    return
                except Exception as e:
                    print(f"  RAGStore: could not load index ({e}), creating fresh.")
            self._index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            # numpy fallback: store raw matrix + id_map in .npy file
            self._index = None
            npy_path = self.store_dir / "embeddings.npy"
            if npy_path.exists() and self._map_path.exists():
                try:
                    self._np_matrix = np.load(str(npy_path))   # (N, D)
                    with open(self._map_path, encoding='utf-8') as f:
                        self._id_map = json.load(f)
                    return
                except Exception as e:
                    print(f"  RAGStore (numpy): could not load embeddings ({e}).")
            self._np_matrix = np.empty((0, self.embedding_dim), dtype=np.float32)

    def _save_index(self):
        if _FAISS and self._index is not None:
            faiss.write_index(self._index, str(self._idx_path))
        else:
            np.save(str(self.store_dir / "embeddings.npy"), self._np_matrix)
        with open(self._map_path, 'w') as f:
            json.dump(self._id_map, f)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        embedding: np.ndarray,
        verdict: str,
        confidence: float,
        metadata: dict = None,
    ) -> int:
        """
        Add an embedding to the store.

        Parameters
        ----------
        embedding  : (embedding_dim,) float32
        verdict    : "AI-Generated" or "REAL"
        confidence : float in [0, 1]
        metadata   : optional extra info dict

        Returns
        -------
        int  SQLite row id of the inserted record
        """
        emb = np.array(embedding, dtype=np.float32).reshape(1, -1)
        if emb.shape[1] != self.embedding_dim:
            # Pad or truncate gracefully
            tmp = np.zeros((1, self.embedding_dim), dtype=np.float32)
            n = min(emb.shape[1], self.embedding_dim)
            tmp[0, :n] = emb[0, :n]
            emb = tmp

        # Persist metadata in SQLite
        conn = sqlite3.connect(str(self._db_path))
        cur  = conn.execute(
            "INSERT INTO cases (verdict, confidence, timestamp, metadata) VALUES (?,?,?,?)",
            (verdict, float(confidence), time.time(), json.dumps(metadata or {})),
        )
        row_id = cur.lastrowid
        conn.commit()
        conn.close()

        # Add to vector index
        if _FAISS and self._index is not None:
            self._index.add(emb)
        else:
            self._np_matrix = np.vstack([self._np_matrix, emb]) \
                if self._np_matrix.shape[0] > 0 else emb

        self._id_map.append(row_id)
        self._save_index()
        return row_id

    def retrieve(self, embedding: np.ndarray, k: int = 5) -> list:
        """
        Retrieve k nearest neighbors.

        Returns list of dicts:
            { verdict, confidence, distance, timestamp, metadata }
        """
        n_stored = self._n_stored()
        if n_stored == 0:
            return []

        emb = np.array(embedding, dtype=np.float32).reshape(1, -1)
        k_actual = min(k, n_stored)

        if _FAISS and self._index is not None:
            distances, indices = self._index.search(emb, k_actual)
            raw_idx  = indices[0].tolist()
            raw_dist = distances[0].tolist()
        else:
            # Brute-force L2 over numpy matrix
            diff  = self._np_matrix - emb           # (N, D)
            dists = (diff ** 2).sum(axis=1)          # (N,)
            order = np.argsort(dists)[:k_actual]
            raw_idx  = order.tolist()
            raw_dist = dists[order].tolist()

        results = []
        conn = sqlite3.connect(str(self._db_path))
        for idx, dist in zip(raw_idx, raw_dist):
            if idx < 0 or idx >= len(self._id_map):
                continue
            row_id = self._id_map[idx]
            row = conn.execute(
                "SELECT verdict, confidence, timestamp, metadata FROM cases WHERE id=?",
                (row_id,),
            ).fetchone()
            if row:
                results.append({
                    "verdict":    row[0],
                    "confidence": float(row[1]),
                    "timestamp":  float(row[2]),
                    "metadata":   json.loads(row[3] or "{}"),
                    "distance":   float(dist),
                })
        conn.close()
        return results

    def augment_probability(
        self,
        model_prob: float,
        neighbors: list,
        weight: float = 0.25,
    ) -> float:
        """
        Blend model probability with a distance-weighted vote from neighbors.

        Parameters
        ----------
        model_prob : original model prediction [0, 1]
        neighbors  : output of retrieve()
        weight     : how much to blend in RAG vote (0 = pure model, 1 = pure RAG)

        Returns
        -------
        float  augmented probability in [0, 1]
        """
        if not neighbors:
            return model_prob

        # Closer neighbors get exponentially higher weight
        votes, weights = [], []
        for n in neighbors:
            d = max(float(n['distance']), 1e-6)
            w = float(np.exp(-d * 0.001))          # scale for high-dim L2 distances
            vote = 1.0 if n['verdict'] == 'AI-Generated' else 0.0
            votes.append(vote * w)
            weights.append(w)

        total_w = sum(weights)
        if total_w == 0:
            return model_prob

        rag_prob = sum(votes) / total_w
        result   = (1.0 - weight) * model_prob + weight * rag_prob
        return float(np.clip(result, 0.0, 1.0))

    def stats(self) -> dict:
        conn = sqlite3.connect(str(self._db_path))
        total    = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        ai_count = conn.execute(
            "SELECT COUNT(*) FROM cases WHERE verdict='AI-Generated'"
        ).fetchone()[0]
        conn.close()
        return {
            "total_cases":    total,
            "ai_cases":       ai_count,
            "real_cases":     total - ai_count,
            "index_vectors":  self._n_stored(),
            "embedding_dim":  self.embedding_dim,
            "faiss_available": _FAISS,
            "store_dir":      str(self.store_dir),
        }

    def clear(self):
        """Remove all stored cases and reset the index."""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("DELETE FROM cases")
        conn.commit()
        conn.close()

        if _FAISS and self._index is not None:
            self._index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            self._np_matrix = np.empty((0, self.embedding_dim), dtype=np.float32)

        self._id_map = []
        self._save_index()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _n_stored(self) -> int:
        if _FAISS and self._index is not None:
            return self._index.ntotal
        return self._np_matrix.shape[0]
