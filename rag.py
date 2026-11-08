"""
rag.py — the question-answering pipeline + a small web API.

Run the API:   uvicorn rag:app --reload      (then open http://localhost:8000/docs)
Quick test:    python rag.py

The pipeline for every question:
  1. dense search  (ChromaDB: meaning)        → 20 chunk ids
  2. BM25 search   (keywords)                 → 20 chunk ids
  3. RRF           (merge the two lists)      → top 10
  4. LLM-as-judge  (score each of the 10)     → best 3
  5. LLM answer    (using ONLY those 3, with citations)
"""

import json
import os
import re
from contextlib import asynccontextmanager
from typing import Literal

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()  # read LLM_BASE_URL, LLM_API_KEY, LLM_MODEL from the .env file

# ── SETTINGS ─────────────────────────────────────────────────────────
CHUNKS_FILE = "data/chunks.json"      # written by ingest.py (BM25 needs the raw text)
CHROMA_DIR = "chroma_db"              # written by ingest.py (dense search)
COLLECTION_NAME = "arxiv_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # MUST be the same model ingest.py used
DENSE_N = 20                          # how many chunks dense search returns
BM25_N = 20                           # how many chunks BM25 returns
FUSED_N = 10                          # how many fused chunks go to the LLM judge
FINAL_N = 3                           # how many chunks the answer is written from
RRF_K = 60                            # the standard RRF constant (explained below)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")

Mode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


# ── LOADING (once) ───────────────────────────────────────────────────
_resources: dict = {}   # filled the first time load_resources() runs, then reused


def tokenize(text: str) -> list[str]:
    """What: lowercase the text and split it into words (letters and digits only).
    Why: BM25 counts matching WORDS, so "LoRA," and "lora" must become the same word."""
    return re.findall(r"[a-z0-9]+", text.lower())


def load_resources() -> dict:
    """What: load the embedding model, open ChromaDB, and build the BM25 index.
    Why: these are slow to load, so do it once and reuse them for every question."""
    if _resources:
        return _resources

    print("Loading chunks, model, ChromaDB and BM25 index...")
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    _resources["model"] = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    _resources["collection"] = client.get_collection(COLLECTION_NAME)

    # Look up any chunk by its id: {"2410.00002v1_1": {...chunk...}, ...}
    _resources["chunks_by_id"] = {chunk["chunk_id"]: chunk for chunk in chunks}

    # BM25 works with positions (0, 1, 2...), so remember which id sits at each position.
    _resources["bm25_ids"] = [chunk["chunk_id"] for chunk in chunks]
    _resources["bm25"] = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])

    print(f"Ready: {len(chunks)} chunks loaded.")
    return _resources


# ── STEP 1: DENSE SEARCH (meaning) ───────────────────────────────────
def dense_search(query: str, n: int = DENSE_N) -> list[str]:
    """What: embed the question and ask ChromaDB for the n closest chunks.
    Why: finds chunks with the same MEANING, even when the words are different."""
    res = load_resources()
    query_embedding = res["model"].encode(query, normalize_embeddings=True).tolist()
    results = res["collection"].query(query_embeddings=[query_embedding], n_results=n)
    return results["ids"][0]          # closest first


# ── STEP 2: BM25 SEARCH (keywords) ───────────────────────────────────
def bm25_search(query: str, n: int = BM25_N) -> list[str]:
    """What: score every chunk by how well its WORDS match the question's words.
    Why: catches exact terms (model names, acronyms) that meaning search can miss."""
    res = load_resources()
    scores = res["bm25"].get_scores(tokenize(query))      # one score per chunk
    positions = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    # Keep only chunks that share at least one word with the question (score > 0).
    return [res["bm25_ids"][i] for i in positions[:n] if scores[i] > 0]    # best first


# ── STEP 3: RECIPROCAL RANK FUSION (merge) ───────────────────────────
def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    """What: merge several ranked lists into one, using only each chunk's RANK.
    Why: BM25 scores and cosine distances are on different scales and can't be
    added directly, but rank 1, rank 2, rank 3... mean the same thing in every list."""
    fused_scores: dict[str, float] = {}
    for ranked_ids in ranked_lists:
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = 0.0
            fused_scores[chunk_id] += 1 / (k + rank)
    return sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)


# ── STEP 4: LLM-AS-JUDGE RERANKING ───────────────────────────────────
def call_llm(prompt: str, max_tokens: int = 300) -> str:
    """What: send one prompt to the LLM and return its reply as text.
    Why: one place for every LLM call. The OpenAI-compatible client works with
    OpenAI, Groq, a local Ollama, etc. Only the .env settings change."""
    if not LLM_API_KEY or not LLM_MODEL:
        raise RuntimeError("Set LLM_API_KEY and LLM_MODEL in your .env file.")
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,                 # 0 = as consistent as possible, no creativity
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def parse_score(reply: str) -> float:
    """What: pull the first number out of the LLM's reply ("8", "Score: 7").
    Why: LLMs don't always reply with only a number. If there's none, score 0."""
    match = re.search(r"\d+(\.\d+)?", reply)
    if match is None:
        return 0.0
    return min(float(match.group()), 10.0)


def llm_judge_rerank(query: str, chunk_ids: list[str], top_n: int = FINAL_N) -> list[str]:
    """What: ask the LLM to rate each candidate chunk 0–10 for relevance, keep the best.
    Why: the LLM reads question and chunk TOGETHER, so it judges relevance more
    carefully than embeddings or keywords. Cost: one LLM call per chunk."""
    res = load_resources()
    scored = []
    for chunk_id in chunk_ids:
        chunk = res["chunks_by_id"][chunk_id]
        prompt = (
            "Rate how relevant this passage is to the question, "
            "from 0 (not relevant) to 10 (perfectly relevant). "
            "Reply with ONLY the number.\n\n"
            f"Question: {query}\n\nPassage: {chunk['text']}"
        )
        scored.append((chunk_id, parse_score(call_llm(prompt, max_tokens=5))))
    # Stable sort: chunks with equal scores keep their RRF order.
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [chunk_id for chunk_id, score in scored[:top_n]]


# ── STEP 5: GENERATE THE ANSWER ──────────────────────────────────────
def generate_answer(query: str, chunk_ids: list[str]) -> str:
    """What: give the LLM the best chunks and ask it to answer ONLY from them, with citations.
    Why: grounding the answer in retrieved text reduces made-up answers (hallucination)."""
    res = load_resources()
    sources = []
    for number, chunk_id in enumerate(chunk_ids, start=1):
        chunk = res["chunks_by_id"][chunk_id]
        sources.append(f"[{number}] {chunk['title']}\n{chunk['text']}")
    prompt = (
        "Answer the question using ONLY the sources below. "
        "Cite the sources you use by their number, like [1]. "
        "If the sources do not contain the answer, say that you don't know.\n\n"
        "Sources:\n" + "\n\n".join(sources) + f"\n\nQuestion: {query}\nAnswer:"
    )
    return call_llm(prompt, max_tokens=400)


# ── THE WHOLE PIPELINE ───────────────────────────────────────────────
def retrieve(query: str, mode: Mode = "hybrid_rerank") -> list[str]:
    """What: find the FINAL_N best chunk ids, using one of four retrieval modes.
    Why: evaluate.py compares the modes to show whether hybrid + reranking actually helps."""
    if mode == "dense":
        return dense_search(query)[:FINAL_N]
    if mode == "bm25":
        return bm25_search(query)[:FINAL_N]
    fused = reciprocal_rank_fusion([dense_search(query), bm25_search(query)])[:FUSED_N]
    if mode == "hybrid":
        return fused[:FINAL_N]
    return llm_judge_rerank(query, fused)


def ask(query: str, mode: Mode = "hybrid_rerank") -> dict:
    """What: retrieve the best chunks, write an answer, and return answer + sources.
    Why: the single entry point used by the API and by evaluate.py."""
    res = load_resources()
    chunk_ids = retrieve(query, mode)
    answer = generate_answer(query, chunk_ids)
    chunks = [res["chunks_by_id"][cid] for cid in chunk_ids]
    return {
        "answer": answer,
        "sources": [{"title": c["title"], "url": c["url"], "paper_id": c["paper_id"]} for c in chunks],
        "contexts": [c["text"] for c in chunks],   # RAGAS needs the exact text used
    }


# ── THE WEB API ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_resources()          # load everything ONCE at startup, not on the first request
    yield


app = FastAPI(title="arXiv RAG", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    mode: Mode = "hybrid_rerank"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
def ask_endpoint(request: AskRequest) -> dict:
    return ask(request.question, request.mode)


if __name__ == "__main__":
    result = ask("How does retrieval-augmented generation reduce hallucination?")
    print("\nANSWER:\n" + result["answer"])
    print("\nSOURCES:")
    for source in result["sources"]:
        print(f"  - {source['title']}  ({source['url']})")
