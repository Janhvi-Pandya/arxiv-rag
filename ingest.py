"""
ingest.py — run ONCE:  python ingest.py

What it does, start to finish:
  1. Download 5,000+ arXiv paper abstracts (ML + NLP).
  2. Cut long abstracts into smaller chunks.
  3. Turn every chunk into an embedding (384 numbers).
  4. Store the chunks + embeddings in ChromaDB (for meaning search).
  5. Save the chunk text to a file (for keyword search later).
"""

import json
import os
import time

import arxiv
import chromadb
from sentence_transformers import SentenceTransformer

# ── SETTINGS ─────────────────────────────────────────────────────────
TARGET_PAPERS = 5200                  # a bit over 5,000, in case a few come back empty
ARXIV_QUERY = "cat:cs.CL OR cat:cs.LG"  # cs.CL = NLP, cs.LG = machine learning
PAPERS_FILE = "data/papers.json"      # raw downloaded papers (so we never download twice)
CHUNKS_FILE = "data/chunks.json"      # every chunk's text (BM25 needs this)
CHROMA_DIR = "chroma_db"              # where ChromaDB saves its files
COLLECTION_NAME = "arxiv_chunks"      # the "table" name inside ChromaDB
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # same small, local model as the HR project
CHUNK_SIZE = 200                      # max words per chunk
CHUNK_OVERLAP = 50                    # words shared between neighbouring chunks
BATCH_SIZE = 500                      # how many chunks we add to ChromaDB at a time


def download_papers() -> list[dict]:
    """What: get abstracts from arXiv, or load them if we already downloaded them.
    Why: downloading takes minutes and arXiv asks us not to hammer it, so do it once."""
    if os.path.exists(PAPERS_FILE):
        print(f"Found {PAPERS_FILE}, loading instead of downloading.")
        with open(PAPERS_FILE, encoding="utf-8") as f:
            return json.load(f)

    print(f"Downloading up to {TARGET_PAPERS} abstracts from arXiv...")
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=5)
    search = arxiv.Search(
        query=ARXIV_QUERY,
        max_results=TARGET_PAPERS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    papers = []
    for result in client.results(search):
        papers.append({
            "id": result.get_short_id(),
            "title": result.title.strip(),
            "abstract": result.summary.replace("\n", " ").strip(),
            "url": result.entry_id,
        })
        if len(papers) % 500 == 0:
            print(f"  downloaded {len(papers)} papers")

    os.makedirs("data", exist_ok=True)
    with open(PAPERS_FILE, "w", encoding="utf-8") as f:
        json.dump(papers, f)
    print(f"Saved {len(papers)} papers to {PAPERS_FILE}")
    return papers


def split_into_chunks(text: str) -> list[str]:
    """What: cut one abstract into pieces of at most CHUNK_SIZE words.
    Why: the embedding model only reads about 256 tokens; anything longer gets ignored."""
    words = text.split()
    if len(words) <= CHUNK_SIZE:
        return [text]                      # most abstracts are short: keep them whole

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP      # move forward 150 words each time
    for start in range(0, len(words), step):
        piece = words[start:start + CHUNK_SIZE]
        chunks.append(" ".join(piece))
        if start + CHUNK_SIZE >= len(words):
            break                          # this piece already reached the end
    return chunks


def build_chunks(papers: list[dict]) -> list[dict]:
    """What: turn every paper into one or more chunk records with an id and metadata.
    Why: search works on chunks, but we must remember which paper each chunk came from."""
    all_chunks = []
    for paper in papers:
        pieces = split_into_chunks(paper["abstract"])
        for number, piece in enumerate(pieces):
            all_chunks.append({
                "chunk_id": f"{paper['id']}_{number}",
                "paper_id": paper["id"],
                "title": paper["title"],
                "url": paper["url"],
                "chunk_number": number,
                "text": piece,
            })
    return all_chunks


def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    """What: turn every chunk's text into 384 numbers.
    Why: similar meaning -> similar numbers, which is what makes meaning search work."""
    print(f"Loading embedding model {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [chunk["text"] for chunk in chunks]
    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return vectors.tolist()


def store_in_chroma(chunks: list[dict], embeddings: list[list[float]]) -> None:
    """What: save chunks + embeddings into ChromaDB on disk.
    Why: ChromaDB can find the closest embeddings fast, without checking every chunk."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Start fresh each run, so running this twice doesn't create duplicates.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # compare embeddings with cosine distance
    )

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=embeddings[start:start + BATCH_SIZE],
            metadatas=[
                {"paper_id": c["paper_id"], "title": c["title"],
                 "url": c["url"], "chunk_number": c["chunk_number"]}
                for c in batch
            ],
        )
        print(f"  stored {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")


def save_chunks_for_bm25(chunks: list[dict]) -> None:
    """What: write every chunk to data/chunks.json.
    Why: BM25 keyword search needs the raw text, and rebuilds its index from this file."""
    os.makedirs("data", exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f)


def main() -> None:
    start_time = time.time()
    papers = download_papers()
    chunks = build_chunks(papers)
    print(f"{len(papers)} papers -> {len(chunks)} chunks")
    embeddings = embed_chunks(chunks)
    store_in_chroma(chunks, embeddings)
    save_chunks_for_bm25(chunks)
    minutes = (time.time() - start_time) / 60
    print(f"Done: {len(papers)} papers, {len(chunks)} chunks, {minutes:.1f} minutes.")


if __name__ == "__main__":
    main()
