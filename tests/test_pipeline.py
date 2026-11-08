"""Tests for chunking, BM25, RRF, score parsing, the 4 retrieval modes, the API and hit rate."""

import pytest
from fastapi.testclient import TestClient

import evaluate
import ingest
import rag

SAMPLE_CHUNKS = [
    {"chunk_id": "p1_0", "paper_id": "p1", "title": "LoRA: Low-Rank Adaptation", "url": "u1",
     "text": "LoRA freezes the pretrained weights and trains small low-rank matrices, making fine-tuning cheap."},
    {"chunk_id": "p2_0", "paper_id": "p2", "title": "RAG for Knowledge Tasks", "url": "u2",
     "text": "Retrieval-augmented generation grounds answers in retrieved documents and reduces hallucination."},
    {"chunk_id": "p3_0", "paper_id": "p3", "title": "Graph Nets for Molecules", "url": "u3",
     "text": "Graph neural networks predict molecular properties from atoms and bonds."},
    {"chunk_id": "p4_0", "paper_id": "p4", "title": "Faithful Summaries", "url": "u4",
     "text": "Language models often invent facts; grounding in sources keeps outputs faithful."},
    {"chunk_id": "p5_0", "paper_id": "p5", "title": "Efficient Fine-Tuning Survey", "url": "u5",
     "text": "Parameter-efficient fine-tuning methods such as adapters and LoRA reduce memory."},
]
FAKE_DENSE = ["p2_0", "p4_0", "p1_0", "p5_0", "p3_0"]   # pretend ChromaDB returned this


@pytest.fixture
def sample_rag(monkeypatch):
    """Fill rag's resource box with 5 sample chunks, and fake the model + LLM."""
    rag._resources.clear()
    rag._resources.update({
        "chunks_by_id": {c["chunk_id"]: c for c in SAMPLE_CHUNKS},
        "bm25_ids": [c["chunk_id"] for c in SAMPLE_CHUNKS],
        "bm25": rag.BM25Okapi([rag.tokenize(c["text"]) for c in SAMPLE_CHUNKS]),
    })
    monkeypatch.setattr(rag, "dense_search", lambda query, n=rag.DENSE_N: FAKE_DENSE[:n])

    def fake_llm(prompt, max_tokens=300):
        if max_tokens == 5:                             # the judge: rate the RAG chunk highest
            return "9" if "hallucination." in prompt else "3"
        return "Fake answer citing [1]."
    monkeypatch.setattr(rag, "call_llm", fake_llm)
    yield
    rag._resources.clear()


# ── ingest.py ────────────────────────────────────────────────────────
def test_short_abstract_stays_whole():
    assert len(ingest.split_into_chunks(" ".join(["x"] * 120))) == 1


def test_long_abstract_is_split_with_overlap():
    words = [f"w{i}" for i in range(420)]
    chunks = ingest.split_into_chunks(" ".join(words))
    assert [len(c.split()) for c in chunks] == [200, 200, 120]
    assert chunks[1].split()[0] == "w150"              # 50 words shared with chunk 1


def test_build_chunks_labels_every_piece():
    papers = [{"id": "A", "title": "T", "url": "u", "abstract": " ".join(["a"] * 420)}]
    chunks = ingest.build_chunks(papers)
    assert [c["chunk_id"] for c in chunks] == ["A_0", "A_1", "A_2"]
    assert all(c["paper_id"] == "A" for c in chunks)


# ── rag.py ───────────────────────────────────────────────────────────
def test_tokenize():
    assert rag.tokenize("LoRA, a Low-Rank method!") == ["lora", "a", "low", "rank", "method"]


def test_bm25_finds_keyword_matches(sample_rag):
    results = rag.bm25_search("What is LoRA fine-tuning?")
    assert results[0] in {"p1_0", "p5_0"}
    assert "p3_0" not in results                       # shares no words with the question


def test_rrf_rewards_chunks_high_in_both_lists():
    dense = ["a", "b", "c"]
    bm25 = ["d", "b", "e"]
    fused = rag.reciprocal_rank_fusion([dense, bm25])
    assert fused[0] == "b"                              # rank 2 in both beats rank 1 in only one
    assert set(fused) == {"a", "b", "c", "d", "e"}


@pytest.mark.parametrize("reply, expected", [
    ("8", 8.0), ("Score: 7/10", 7.0), ("9.5", 9.5), ("I'd say about 12", 10.0), ("no idea", 0.0),
])
def test_parse_score(reply, expected):
    assert rag.parse_score(reply) == expected


def test_four_modes(sample_rag):
    q = "how does RAG reduce hallucination"
    assert rag.retrieve(q, "dense") == FAKE_DENSE[:3]
    assert len(rag.retrieve(q, "bm25")) <= 3
    assert len(rag.retrieve(q, "hybrid")) == 3
    assert rag.retrieve(q, "hybrid_rerank")[0] == "p2_0"   # the judge's favourite comes first


def test_api(sample_rag, monkeypatch):
    monkeypatch.setattr(rag, "load_resources", lambda: rag._resources)
    client = TestClient(rag.app)
    assert client.get("/health").json() == {"status": "ok"}

    r = client.post("/ask", json={"question": "how does RAG reduce hallucination", "mode": "hybrid"})
    assert r.status_code == 200
    assert r.json()["answer"] == "Fake answer citing [1]."
    assert len(r.json()["sources"]) == 3

    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "hi", "mode": "magic"}).status_code == 422


def test_space_home_redirects_to_docs(sample_rag, monkeypatch):
    monkeypatch.setattr(rag, "load_resources", lambda: rag._resources)
    import space_app
    r = TestClient(space_app.app).get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/docs"


# ── evaluate.py ──────────────────────────────────────────────────────
def test_hit_rate(sample_rag):
    questions = [
        {"question": "how does RAG reduce hallucination", "ground_truth": "-", "source_paper_id": "p2"},
        {"question": "graph networks for molecules", "ground_truth": "-", "source_paper_id": "p3"},
    ]
    # dense returns p2, p4, p1 for every question: first is a hit, second a miss
    assert evaluate.hit_rate(questions, "dense") == 0.5
