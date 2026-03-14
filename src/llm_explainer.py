"""
llm_explainer.py — RAG-based LLM explanation for AI detection results.

Knowledge base  : data/rag_knowledge/*.txt
Retriever       : TF-IDF cosine similarity (sklearn, already installed)
LLM backends    : 1. Ollama  — local HTTP (default: http://localhost:11434)
                  2. OpenAI  — OPENAI_API_KEY env var
                  3. Template — no LLM dependency (always available)

CAG (KV-cache equivalent)
--------------------------
On startup, the full system prompt + base knowledge is built once and stored
in `_warmed_context`.  Every explain() call reuses this exact string as the
system message so that any LLM backend (Ollama, OpenAI) can cache the
prefix key-value state and avoid re-encoding it on every request.

Environment variables
---------------------
OLLAMA_URL   : base URL for Ollama  (default: http://localhost:11434)
OLLAMA_MODEL : model name to use    (default: llama3.2)
OPENAI_API_KEY : enables OpenAI backend
"""

import json
import os
import re
import urllib.request
import urllib.error
from pathlib import Path

_KB_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'rag_knowledge'
)
_OLLAMA_URL   = os.environ.get("OLLAMA_URL",   "http://localhost:11434")
_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

# How many knowledge chunks to include in the pre-warmed context
_CONTEXT_CHUNKS = 6
# How many chunks to retrieve per query
_RETRIEVE_K     = 3
# Max tokens to request from the LLM
_MAX_TOKENS     = 220


class LLMExplainer:
    """
    RAG + LLM explanations for AI detection results.

    Usage
    -----
    explainer = LLMExplainer()
    text = explainer.explain(detection_result, neighbors=[...])
    """

    def __init__(self):
        self._chunks: list    = []
        self._vectorizer      = None
        self._tfidf_matrix    = None
        self._backend: str    = 'template'
        self._ollama_model    = _OLLAMA_MODEL
        self._warmed_context  = ''    # CAG: pre-built system prompt

        self._load_knowledge_base()
        self._build_retriever()
        self._detect_backend()
        self._warm_context()            # CAG: build once on startup

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _load_knowledge_base(self):
        kb = Path(_KB_DIR)
        if not kb.exists():
            return
        for f in sorted(kb.glob("*.txt")):
            text   = f.read_text(encoding='utf-8')
            chunks = [c.strip() for c in re.split(r'\n{2,}', text) if len(c.strip()) > 40]
            self._chunks.extend(chunks)

    def _build_retriever(self):
        """TF-IDF retriever over knowledge chunks (sklearn is available)."""
        if not self._chunks:
            return
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            self._vectorizer   = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
            self._tfidf_matrix = self._vectorizer.fit_transform(self._chunks)
            self._cos_sim      = cosine_similarity
        except Exception as e:
            print(f"  LLMExplainer: TF-IDF build failed ({e}), using first-N fallback.")

    def _detect_backend(self):
        """Probe available LLM backends (Ollama → OpenAI → template)."""
        # 1. Ollama
        try:
            req = urllib.request.Request(
                f"{_OLLAMA_URL}/api/tags", method="GET"
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                data   = json.loads(resp.read())
                models = [m['name'] for m in data.get('models', [])]
                if models:
                    self._backend = 'ollama'
                    # If preferred model not present, pick first available
                    if not any(_OLLAMA_MODEL in m for m in models):
                        self._ollama_model = models[0].split(':')[0]
                    return
        except Exception:
            pass

        # 2. OpenAI
        if os.environ.get("OPENAI_API_KEY"):
            self._backend = 'openai'
            return

        # 3. Template fallback
        self._backend = 'template'

    def _warm_context(self):
        """
        CAG: build the system prompt + base knowledge once.

        This string is reused verbatim as the `system` role message for
        every LLM call, allowing Ollama / OpenAI to cache the KV-state
        for the shared prefix and only encode the per-image user prompt.
        """
        base_chunks = self._chunks[:_CONTEXT_CHUNKS]
        kb_text     = "\n\n".join(base_chunks) if base_chunks else ""

        intro = (
            "You are an expert in digital forensics and AI-generated image detection. "
            "Your task is to explain detection results clearly, citing specific forensic "
            "evidence. Be concise (3–5 sentences), factual, and avoid speculation."
        )

        self._warmed_context = (
            f"{intro}\n\nBackground knowledge:\n{kb_text}"
            if kb_text else intro
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve(self, query: str, k: int = _RETRIEVE_K) -> list:
        if not self._chunks:
            return []
        if self._vectorizer is None:
            return self._chunks[:k]
        try:
            q_vec = self._vectorizer.transform([query])
            sims  = self._cos_sim(q_vec, self._tfidf_matrix)[0]
            idx   = sims.argsort()[::-1][:k]
            return [self._chunks[i] for i in idx if sims[i] > 0.01]
        except Exception:
            return self._chunks[:k]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(self, detection_result: dict, neighbors: list = None) -> str:
        """
        Generate a natural language explanation for a detection result.

        Parameters
        ----------
        detection_result : dict from Detector.predict()
        neighbors        : list from ImageRAGStore.retrieve() (optional)

        Returns
        -------
        str  explanation text
        """
        ai_prob    = detection_result.get("ai_probability", 0.0)
        conclusion = detection_result.get("conclusion", "Unknown")
        prnu       = detection_result.get("prnu_analysis", {})
        model_type = detection_result.get("model_type", "unknown model")
        platform   = detection_result.get("platform_compression")

        # --- Build retrieval query ---
        q_parts = [f"{conclusion} image detection forensics"]
        if ai_prob > 0.8:
            q_parts.append("high confidence AI generation artifacts GAN diffusion")
        elif ai_prob > 0.5:
            q_parts.append("moderate AI probability PRNU noise frequency analysis")
        else:
            q_parts.append("real camera noise PRNU fingerprint natural statistics")
        if platform:
            q_parts.append(f"{platform.get('codec','')} compression platform artifacts")

        relevant = self._retrieve(" ".join(q_parts))

        # --- Build user prompt ---
        lines = [
            "Detection result to explain:",
            f"  Verdict:          {conclusion}",
            f"  AI Probability:   {ai_prob*100:.1f}%",
            f"  Model:            {model_type}",
        ]

        noise_str = prnu.get("noise_strength")
        if noise_str is not None:
            lines.append(f"  PRNU Noise Strength: {noise_str}")

        if platform:
            lines.append(
                f"  Platform: {platform.get('platform','?')} "
                f"({platform.get('codec','?')}, "
                f"PRNU reliability {platform.get('prnu_reliability','?')})"
            )

        if neighbors:
            similar = [
                f"{n['verdict']} ({n['confidence']*100:.0f}%)"
                for n in neighbors[:3]
            ]
            lines.append(f"  Similar past cases: {', '.join(similar)}")

        if relevant:
            lines.append("\nRelevant forensic knowledge:")
            for chunk in relevant:
                lines.append(f"  • {chunk[:250]}")

        lines.append(
            f"\nWrite a clear 3-5 sentence explanation of why this image was "
            f"classified as '{conclusion}', citing the specific evidence above."
        )

        prompt = "\n".join(lines)

        # --- Dispatch to backend ---
        if self._backend == 'ollama':
            return self._call_ollama(prompt)
        if self._backend == 'openai':
            return self._call_openai(prompt)
        return self._template(detection_result, neighbors)

    # ------------------------------------------------------------------
    # LLM backends
    # ------------------------------------------------------------------

    def _call_ollama(self, user_prompt: str) -> str:
        """Call Ollama chat API, reusing the pre-warmed system context."""
        try:
            payload = json.dumps({
                "model": self._ollama_model,
                "messages": [
                    {"role": "system", "content": self._warmed_context},
                    {"role": "user",   "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": _MAX_TOKENS},
            }).encode()

            req = urllib.request.Request(
                f"{_OLLAMA_URL}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                return data.get("message", {}).get("content", "").strip()
        except Exception as e:
            return f"[Ollama error: {e}]"

    def _call_openai(self, user_prompt: str) -> str:
        """Call OpenAI chat API, reusing the pre-warmed system context."""
        try:
            import openai
            client   = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self._warmed_context},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=_MAX_TOKENS,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[OpenAI error: {e}]"

    # ------------------------------------------------------------------
    # Template fallback (no LLM)
    # ------------------------------------------------------------------

    def _template(self, result: dict, neighbors: list | None) -> str:
        ai_prob    = result.get("ai_probability", 0.0)
        conclusion = result.get("conclusion", "Unknown")
        prnu       = result.get("prnu_analysis", {})

        conf_word = (
            "very high" if ai_prob > 0.85 or ai_prob < 0.15
            else "high" if ai_prob > 0.72 or ai_prob < 0.28
            else "moderate"
        )

        if conclusion == "AI-Generated":
            pct  = ai_prob * 100
            text = (
                f"This image was classified as AI-Generated with {conf_word} confidence "
                f"({pct:.1f}% AI probability). "
            )
            ns = prnu.get("noise_strength")
            if ns is not None:
                try:
                    if float(ns) < 0.008:
                        text += (
                            "The PRNU noise analysis found an unusually smooth noise floor "
                            "inconsistent with any real camera sensor fingerprint. "
                        )
                    else:
                        text += (
                            "Multi-branch forensic analysis detected statistical anomalies "
                            "in the frequency spectrum and noise distribution. "
                        )
                except (TypeError, ValueError):
                    pass
            if neighbors:
                ai_n = sum(1 for n in neighbors if n["verdict"] == "AI-Generated")
                text += (
                    f"{ai_n} of the {len(neighbors)} most similar previously analyzed "
                    "images were also classified as AI-generated, reinforcing this verdict."
                )
        else:
            pct  = (1.0 - ai_prob) * 100
            text = (
                f"This image was classified as REAL with {conf_word} confidence "
                f"({pct:.1f}% real probability). "
                "The forensic analysis found noise patterns consistent with authentic "
                "camera sensor characteristics. "
            )
            if neighbors:
                real_n = sum(1 for n in neighbors if n["verdict"] == "REAL")
                text += (
                    f"{real_n} of the {len(neighbors)} most similar previously analyzed "
                    "images were also classified as real."
                )
        return text

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def info(self) -> dict:
        return {
            "backend":         self._backend,
            "ollama_model":    self._ollama_model if self._backend == 'ollama' else None,
            "ollama_url":      _OLLAMA_URL,
            "knowledge_chunks": len(self._chunks),
            "retriever":       "tfidf" if self._vectorizer else "first-N",
            "context_warmed":  bool(self._warmed_context),
            "context_len":     len(self._warmed_context),
        }
