---
title: arXiv RAG
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# arXiv RAG: hybrid search question answering over ML research papers

Ask a question about recent machine learning or NLP research and get an answer written only from real arXiv paper abstracts, with citations. The system finds relevant text with two kinds of search (meaning-based and keyword-based), merges them, lets an LLM rerank the results, and then writes a grounded answer. It is built from scratch, without LangChain, so every step is visible.

## Architecture

```
ONCE (ingest.py)
  arXiv API ──► ~5,200 abstracts ──► chunks (200 words, 50 overlap)
                                        ├──► embeddings (all-MiniLM-L6-v2) ──► ChromaDB
                                        └──► data/chunks.json (for BM25)

EVERY QUESTION (rag.py,  POST /ask)
  question ─┬─► dense search (ChromaDB, meaning)  → 20 ─┐
            └─► BM25 search  (keywords)            → 20 ─┴─► RRF merge → top 10
                                                               │
                                              LLM-as-judge scores each (0–10) → best 3
                                                               │
                                              LLM answers ONLY from those 3, with [1] citations

MEASURE (evaluate.py)
  20 test questions ──► hit rate for dense / bm25 / hybrid / hybrid_rerank
                    └──► RAGAS: faithfulness, answer relevancy, context recall
```

## Files

| File | What it does |
|---|---|
| `ingest.py` | run once: download abstracts, chunk, embed, store in ChromaDB, save chunks for BM25 |
| `rag.py` | the pipeline (dense, BM25, RRF, LLM judge, answer) and the FastAPI `/ask` endpoint |
| `evaluate.py` | makes test questions, then measures hit rate and RAGAS scores |
| `space_app.py` | entry point for the deployed app (sends the home page to `/docs`) |
| `tests/` | tests for chunking, BM25, RRF, score parsing, the 4 modes, the API and hit rate |
| `Dockerfile` | builds the deployed app, including the index |

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # small CPU-only PyTorch
pip install -r requirements.txt
cp .env.example .env          # then put your LLM key in .env (Groq's free tier works)

python ingest.py              # once: a few minutes (arXiv asks for a 3 s pause per 100 papers)
uvicorn rag:app --reload      # then open http://localhost:8000/docs and try POST /ask
```

Example request:

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "How does LoRA make fine-tuning cheaper?", "mode": "hybrid_rerank"}'
```

`mode` can be `dense`, `bm25`, `hybrid` or `hybrid_rerank` (the default).

Tests: `pip install pytest httpx && pytest -v`

## Evaluate it

```bash
pip install -r requirements-eval.txt
python evaluate.py --make-questions   # writes eval_questions.json: read it and fix bad questions
python evaluate.py --skip-ragas       # hit rate only
python evaluate.py                    # hit rate + RAGAS (many LLM calls, a few minutes)
```

Each test question is written from one known paper, so hit rate can check whether that paper came back in the top 3.

## How each part works

**Chunking.** The embedding model only reads about 256 tokens, so abstracts longer than 200 words are cut into 200-word pieces that overlap by 50 words. That way an idea sitting on a cut still appears whole in one piece. Most abstracts are shorter than that and stay as one chunk.

**Embeddings.** `all-MiniLM-L6-v2` turns each chunk into 384 numbers so that texts with similar meaning get similar numbers. They are normalised to length 1, so cosine similarity is just a dot product. It runs locally and is free.

**BM25.** Keyword search that scores chunks by how many of the question's words they contain, giving more weight to rare words like "LoRA" than common ones like "the". It catches exact model names and acronyms that meaning search can miss.

**RRF (Reciprocal Rank Fusion).** BM25 scores and cosine distances are on different scales, so they can't be added. RRF uses only the rank instead: each chunk gets `1 / (60 + rank)` from each list, summed. Chunks that rank well in both lists rise to the top.

**LLM-as-judge.** The LLM reads the question and each of the top 10 chunks together and rates relevance from 0 to 10. This is more careful than either search, but costs one LLM call per chunk, so it only runs on 10 candidates (the cheap searches go first, like a funnel).

**RAGAS metrics.** RAGAS uses an LLM as a grader. Faithfulness checks that the answer's claims are supported by the retrieved chunks. Answer relevancy checks that the answer addresses the question. Context recall checks that the retrieved chunks contain what the correct answer needs.

## Results

Fill these in from your own runs of `ingest.py` and `evaluate.py`. Do not copy numbers from anywhere else.

| | Value |
|---|---|
| Papers downloaded | _ |
| Chunks stored | _ |
| Ingest time | _ |

| Retrieval mode | Hit rate (top 3) |
|---|---|
| dense | _ |
| bm25 | _ |
| hybrid (RRF) | _ |
| hybrid + rerank | _ |

| RAGAS metric | Score |
|---|---|
| faithfulness | _ |
| answer relevancy | _ |
| context recall | _ |

## Deployment (free)

The app runs on a free Hugging Face Space (CPU basic: 2 vCPU, 16 GB RAM). On every push to `main`, the GitHub Actions workflow runs the tests and, if they pass, pushes the repo to the Space, which rebuilds the Docker image. The index is built during the image build, so the running app starts quickly.

One-time setup:

1. Create a Hugging Face account, then a new Space: SDK **Docker**, template **Blank**, hardware **CPU basic (free)**.
2. In the Space's **Settings → Variables and secrets**, add secrets `LLM_API_KEY` (your Groq key), and variables `LLM_BASE_URL=https://api.groq.com/openai/v1` and `LLM_MODEL=llama-3.1-8b-instant`.
3. Create a Hugging Face access token with **write** permission.
4. In this GitHub repo's **Settings → Secrets and variables → Actions**, add the secret `HF_TOKEN`, and the variables `HF_USERNAME` (your Hugging Face username) and `HF_SPACE` (the Space name).
5. Push to `main`, or rerun the workflow from the Actions tab.

The live API docs are then at `https://<HF_USERNAME>-<HF_SPACE>.hf.space/docs`.

## Known limitations and what I'd improve

- **Abstracts only**, not full papers, so answers can't go deeper than what an abstract says.
- **LLM reranking is slow and costs calls**: 10 judge calls per question. On a free tier with a per-minute limit, `hybrid_rerank` handles only a few questions a minute. A small cross-encoder reranker would be faster and free.
- **Small evaluation set** (20 questions, written by an LLM from the abstracts), so the numbers are a rough signal, not a benchmark.
- **The corpus changes with each build**: ingest downloads the newest papers, so a rebuild means a different set of papers.
- **BM25 is rebuilt in memory** at every startup, which is fine at this size but wouldn't scale to millions of chunks.
- **Free Space sleeps** after a period of no visits and takes a little while to wake up.
