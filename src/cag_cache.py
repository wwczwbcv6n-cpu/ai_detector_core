"""
cag_cache.py — CAG (Cache-Augmented Generation) detection cache.

Uses MD5 hashing (fast, no extra dependency) for exact-match lookup.
In-memory OrderedDict LRU + optional SQLite persistence across restarts.

Heatmap data is stripped before caching (too large).
Cache hits include `cache_hit: true` and `cache_age_s` in the response.
"""

import hashlib
import io
import json
import os
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path

_DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'cag_cache'
)
_MAX_MEMORY    = 1000               # max in-memory LRU entries
_DEFAULT_TTL   = 7 * 24 * 3600     # 7 days in seconds

# Keys to strip from cached results (large or ephemeral data)
_STRIP_KEYS = ('heatmap_base64',)


class DetectionCache:
    """
    CAG cache: stores detection results keyed by image MD5 hash.

    Usage
    -----
    cache = DetectionCache()

    cached = cache.lookup(image_bytes)
    if cached is not None:
        return cached          # instant response

    result = detector.predict(image_bytes, ...)
    cache.store(image_bytes, result)
    return result
    """

    def __init__(
        self,
        cache_dir: str = _DEFAULT_CACHE_DIR,
        max_memory: int = _MAX_MEMORY,
        ttl: float = _DEFAULT_TTL,
        persist: bool = True,
    ):
        self._max_memory = max_memory
        self._ttl        = ttl
        self._persist    = persist
        self._mem: OrderedDict = OrderedDict()   # hash → (result_dict, timestamp)

        if persist:
            self._db_path = Path(cache_dir) / "cache.db"
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            self._init_db()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_db(self):
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                img_hash  TEXT PRIMARY KEY,
                result    TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(image_bytes: bytes) -> str:
        """MD5 of raw image bytes — fast and dependency-free."""
        return hashlib.md5(image_bytes).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, image_bytes: bytes) -> dict | None:
        """
        Look up a cached result.

        Returns None on miss or expired entry.
        On hit, returns the result dict with extra fields:
            cache_hit:    True
            cache_age_s:  seconds since the result was stored
        """
        h   = self._hash(image_bytes)
        now = time.time()

        # 1. Memory LRU
        if h in self._mem:
            result, ts = self._mem[h]
            if now - ts < self._ttl:
                self._mem.move_to_end(h)
                return {**result, 'cache_hit': True, 'cache_age_s': round(now - ts, 1)}
            del self._mem[h]

        # 2. SQLite
        if self._persist:
            conn = sqlite3.connect(str(self._db_path))
            row  = conn.execute(
                "SELECT result, timestamp FROM cache WHERE img_hash=?", (h,)
            ).fetchone()
            conn.close()
            if row:
                ts = float(row[1])
                if now - ts < self._ttl:
                    result = json.loads(row[0])
                    self._mem_put(h, result, ts)     # warm memory
                    return {**result, 'cache_hit': True, 'cache_age_s': round(now - ts, 1)}

        return None

    def store(self, image_bytes: bytes, result: dict):
        """
        Cache a detection result.

        Heatmap data is automatically stripped (too large to cache).
        """
        h         = self._hash(image_bytes)
        now       = time.time()
        storable  = {k: v for k, v in result.items() if k not in _STRIP_KEYS}

        self._mem_put(h, storable, now)

        if self._persist:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute(
                "INSERT OR REPLACE INTO cache (img_hash, result, timestamp) VALUES (?,?,?)",
                (h, json.dumps(storable), now),
            )
            conn.commit()
            conn.close()

    def stats(self) -> dict:
        db_count = 0
        if self._persist:
            conn     = sqlite3.connect(str(self._db_path))
            db_count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            conn.close()
        return {
            "memory_entries": len(self._mem),
            "db_entries":     db_count,
            "max_memory":     self._max_memory,
            "ttl_days":       round(self._ttl / 86400, 1),
        }

    def clear(self):
        """Remove all cached entries."""
        self._mem.clear()
        if self._persist:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute("DELETE FROM cache")
            conn.commit()
            conn.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _mem_put(self, h: str, result: dict, ts: float):
        """Insert into LRU memory, evicting oldest if full."""
        if h in self._mem:
            self._mem.move_to_end(h)
        self._mem[h] = (result, ts)
        while len(self._mem) > self._max_memory:
            self._mem.popitem(last=False)
