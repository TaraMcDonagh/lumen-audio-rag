"""
Lumen Audio — grounded Q&A over a knowledge base.

This skeleton runs as-is and returns a stub response. Your job is to fill in the
RAG logic where marked TODO:
  1. chunk + index the documents in data/
  2. retrieve the most relevant chunk(s) for a question
  3. generate a grounded answer WITH a citation to the source file
  4. decline gracefully when the question isn't covered by the KB

Helpers for embeddings and chat completion are in llm.py.
Run with:  python app.py
"""

import os
import glob
from flask import Flask, request, jsonify

import llm  # embed() and complete() — see llm.py

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_documents():
    """Load every markdown article in data/ as (source_name, text)."""
    docs = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.md"))):
        with open(path, "r", encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read().strip()))
    return docs


# Load once at startup. The candidate decides how to chunk/index from here.
DOCUMENTS = load_documents()

# TODO: build your index here (chunk the documents, embed them with llm.embed(),
# and keep the vectors in memory — numpy is fine).


@app.route("/health")
def health():
    return jsonify({"ok": True, "documents_loaded": len(DOCUMENTS)})


@app.route("/ask", methods=["POST"])
def ask():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "send JSON like {\"question\": \"...\"}"}), 400

    # ------------------------------------------------------------------
    # TODO: replace this stub with real retrieval-augmented generation.
    #   - retrieve the most relevant chunk(s) for `question`
    #   - if nothing is relevant enough, return an "I don't know" response
    #   - otherwise call llm.complete(...) to answer using ONLY the retrieved
    #     context, and return the source filename as the citation
    # ------------------------------------------------------------------
    return jsonify({
        "answer": "Not implemented yet — this is the stub.",
        "sources": [],
        "available_sources": [name for name, _ in DOCUMENTS],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
