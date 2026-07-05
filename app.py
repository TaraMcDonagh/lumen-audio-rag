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
from collections import Counter
from math import sqrt
import re
import llm  # embed() and complete() — see llm.py

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REFUSAL = "I don't have enough information in the Lumen Audio knowledge base to answer that."
MIN_RELEVANCE = 0.14
MAX_CONTEXT_CHUNKS = 2
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "have",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "our",
    "the",
    "their",
    "to",
    "what",
    "when",
    "where",
    "with",
    "you",
    "your",
}

TOKEN_ALIASES = {
    "amended": "amend",
    "cancelling": "cancel",
    "cancelled": "cancel",
    "cancellation": "cancel",
    "covering": "cover",
    "deletion": "delete",
    "deleted": "delete",
    "items": "item",
    "orders": "order",
    "refunds": "refund",
    "refunded": "refund",
    "returned": "return",
    "returns": "return",
    "shipping": "ship",
    "shipped": "ship",
    "ships": "ship",
}

def load_documents():
    """Load every markdown article in data/ as (source_name, text)."""
    docs = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.md"))):
        with open(path, "r", encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read().strip()))
    return docs


def normalize_token(token):
    return TOKEN_ALIASES.get(token, token)

def tokenize(text):
    return [
        normalize_token(token)
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]

def vectorise(text):
    return Counter(tokenize(text))

def cosine_similarity(a, b):
    common_words = set(a) & set(b)
    dot_product = sum(a[word] * b[word] for word in common_words)
    a_length = sqrt(sum(value * value for value in a.values()))
    b_length = sqrt(sum(value * value for value in b.values()))

    if a_length == 0 or b_length == 0:
        return 0

    return dot_product / (a_length * b_length)

def chunk_document(source, text):
    """Split a markdown document into small chunks, keeping the title as context."""
    title = source
    body_parts = []
    current_part = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
        elif not stripped and current_part:
            body_parts.append(" ".join(current_part))
            current_part = []
        elif stripped:
            current_part.append(stripped)

    if current_part:
        body_parts.append(" ".join(current_part))

    paragraphs = body_parts or [text]

    return [
        {
            "source": source,
            "chunk_id": f"{source}:{index}",
            "text": f"{title}\n\n{paragraph}",
            "display_text": paragraph,
            "vector": vectorise(f"{title} {paragraph}"),
        }
        for index, paragraph in enumerate(paragraphs)
    ]

# Load and index once at startup.
DOCUMENTS = load_documents()

INDEX = [
    chunk
    for source, text in DOCUMENTS
    for chunk in chunk_document(source, text)
]

def retrieve_chunks(question):
    question_vector = vectorise(question)
    scored_chunks = [
        {
            "source": item["source"],
            "text": item["text"],
            "display_text": item["display_text"],
            "score": cosine_similarity(question_vector, item["vector"]),
        }
        for item in INDEX
    ]
    scored_chunks.sort(key=lambda item: item["score"], reverse=True)
    relevant = [
        item
        for item in scored_chunks[:MAX_CONTEXT_CHUNKS]
        if item["score"] >= MIN_RELEVANCE
    ]

    return relevant

def clean_answer(text):
    text = re.sub(r"^#\s*", "", text)
    return text.strip()

def sentence_score(question, sentence):
    score = cosine_similarity(vectorise(question), vectorise(sentence))
    question_lower = question.lower()
    sentence_lower = sentence.lower()

    if "where" in question_lower and "ship" in question_lower and "ships to" in sentence_lower:
        score += 0.5
    if "cover" in question_lower and "not" not in question_lower and "does not cover" in sentence_lower:
        score -= 0.2
    if "not cover" in question_lower and "does not cover" in sentence_lower:
        score += 0.2

    return score

def best_context_sentence(question, chunks):
    candidates = []

    for chunk in chunks:
        for sentence in re.split(r"(?<=[.!?])\s+", chunk["display_text"]):
            sentence = sentence.strip()
            if sentence:
                candidates.append((
                    sentence_score(question, sentence),
                    sentence,
                    chunk["source"],
                ))

    if not candidates:
        return clean_answer(chunks[0]["display_text"]), [chunks[0]["source"]]

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], [candidates[0][2]]

def answer_from_context(question, chunks):
    context = "\n\n".join(
        f"Source: {chunk['source']}\n{chunk['display_text']}"
        for chunk in chunks
    )

    if os.environ.get("LLM_API_KEY"):
        system = (
            "You answer customer questions for Lumen Audio. Use only the provided "
            "knowledge-base context. If the context does not answer the question, "
            f"reply exactly: {REFUSAL}"
        )
        user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        try:
            answer = llm.complete(system, user).strip()
            if answer and REFUSAL not in answer:
                return answer, sorted({chunk["source"] for chunk in chunks})
        except Exception:
            pass

    return best_context_sentence(question, chunks)

@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "documents_loaded": len(DOCUMENTS),
        "chunks_indexed": len(INDEX),
    })


@app.route("/ask", methods=["POST"])
def ask():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "send JSON like {\"question\": \"...\"}"}), 400

    chunks = retrieve_chunks(question)

    if not chunks:
        return jsonify({
            "answer": REFUSAL,
            "sources": [],
        })

    answer, sources = answer_from_context(question, chunks)

    return jsonify({
        "answer": clean_answer(answer),
        "sources": sources,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
