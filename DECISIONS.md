# Decisions

A short, honest record of how you approached this. Bullet points are fine — this doesn't
need to be long, it needs to be real. It's the backbone of our conversation.

## Assumptions

The brief is deliberately under-specified. What did you assume, and why?
(e.g. who the user is, what "grounded" should mean, how strict the refusal should be.)

- The endpoint is for customer-support style answers, so short, direct answers are better
  than long summaries.
- "Grounded" means the answer must be recoverable from one or more `data/*.md` files and
  the response must include those source filenames.
- Refusal should be conservative: if retrieval cannot find a meaningful match, the API
  should return the fixed refusal and no sources rather than guessing.

## What I built and prioritised

What did you build first, and what does the working slice do?

- I built a small in-memory retrieval pipeline: load the markdown files, chunk them by
  article paragraph, index them with token counters, and rank chunks with cosine
  similarity.
- I fixed the startup indexing bug by using the actual `(source, text)` tuples returned by
  `load_documents()`.
- `/ask` now retrieves relevant chunks, refuses out-of-KB questions, and returns answers
  with `sources` matching the README contract.
- If `LLM_API_KEY` is configured, the app asks the chat model to answer using only the
  retrieved context. Without a key, it falls back to the best matching KB sentence so the
  smoke test and core behavior still work locally.

## What I cut, and why

What did you consciously leave out given the ~2-hour timebox?

- I did not add a vector database or persistent index. The KB is six tiny files, so an
  in-memory index is simpler and easier to defend.
- I did not require hosted embeddings. A lexical retriever is good enough for this toy KB
  and avoids making the app unusable without API credentials.
- I kept chunking simple. For larger or messier docs, I would use token-aware chunking with
  overlap.

## How I'd know it works

How would you evaluate this — beyond "it ran"? What would you measure?

- Run the included smoke test for health, a covered return-window question, and an
  unsupported student-discount question.
- Add a small labeled set of questions for every KB article and measure source accuracy,
  refusal precision, and refusal recall.
- Review answers for faithfulness: every claim in the answer should be traceable to the
  returned source text.

## With more time / to take it to production

What are the next things you'd do, and what would change to run this for real
(multiple clients, real volume, reliability, cost)?

- Swap the lexical index for embeddings plus a real vector store once the KB grows.
- Add structured evaluation and regression tests for retrieval, citation accuracy, and
  refusal behavior.
- Return richer citation metadata, such as chunk IDs or article titles, not only filenames.
- Add observability around latency, model errors, refusal rates, and low-confidence queries.
- Make the LLM path fail closed: if generation fails in production, return a controlled
  fallback rather than a partial or uncited answer.
