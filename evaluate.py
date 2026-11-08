"""
evaluate.py — measure how well the pipeline works.

Step 0 (once):  python evaluate.py --make-questions
                  picks 20 random papers from data/papers.json and asks the LLM to write
                  one question + correct answer from each abstract -> eval_questions.json
                  (read them, and fix or replace any bad ones by hand!)
Step 1:         python evaluate.py
                  hit rate for all 4 retrieval modes + RAGAS on the full pipeline
                python evaluate.py --skip-ragas      (hit rate only, far fewer LLM calls)

Needs the eval extras:  pip install -r requirements-eval.txt
"""

import argparse
import json
import os
import random

import rag

QUESTIONS_FILE = "eval_questions.json"
PAPERS_FILE = "data/papers.json"
RESULTS_FILE = "results/eval_results.json"
NUM_QUESTIONS = 20
MODES = ["dense", "bm25", "hybrid", "hybrid_rerank"]


# ── STEP 0: MAKE THE TEST QUESTIONS ──────────────────────────────────
def make_questions(n: int = NUM_QUESTIONS, seed: int = 42) -> None:
    """What: write n test questions, each one taken from ONE known paper.
    Why: hit rate needs to know which paper is the right answer. Because the question
    is written from that paper's abstract, we know its correct source_paper_id."""
    with open(PAPERS_FILE, encoding="utf-8") as f:
        papers = json.load(f)
    random.seed(seed)                       # same seed -> same papers picked every time
    chosen = random.sample(papers, n)

    questions = []
    for paper in chosen:
        prompt = (
            "Read this research paper abstract. Write ONE specific question that this "
            "abstract answers, and the correct answer in 1-2 sentences, using only the abstract. "
            "Reply in exactly this format:\nQUESTION: ...\nANSWER: ...\n\n"
            f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
        )
        reply = rag.call_llm(prompt, max_tokens=200)
        if "QUESTION:" not in reply or "ANSWER:" not in reply:
            print(f"  skipped {paper['id']} (LLM reply not in the expected format)")
            continue
        question = reply.split("QUESTION:", 1)[1].split("ANSWER:", 1)[0].strip()
        answer = reply.split("ANSWER:", 1)[1].strip()
        questions.append({
            "question": question,
            "ground_truth": answer,
            "source_paper_id": paper["id"],
        })
        print(f"  {len(questions)}/{n}: {question}")

    with open(QUESTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)
    print(f"Saved {len(questions)} questions to {QUESTIONS_FILE}. Read them before trusting them!")


def load_questions() -> list[dict]:
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        return json.load(f)


# ── STEP 1: HIT RATE (no library) ────────────────────────────────────
def hit_rate(questions: list[dict], mode: str) -> float:
    """What: % of questions where the right paper is among the FINAL_N retrieved chunks.
    Why: easy to understand, and it checks RETRIEVAL on its own, without the answer step.
    (hybrid_rerank still calls the LLM judge; the other three modes need no LLM at all.)"""
    res = rag.load_resources()
    hits = 0
    for q in questions:
        chunk_ids = rag.retrieve(q["question"], mode)
        paper_ids = [res["chunks_by_id"][cid]["paper_id"] for cid in chunk_ids]
        if q["source_paper_id"] in paper_ids:
            hits += 1
    return hits / len(questions)


# ── STEP 2: RAGAS (LLM as a grader) ──────────────────────────────────
def run_ragas(questions: list[dict]) -> dict:
    """What: run the full pipeline on every question, then let RAGAS grade the answers.
    Why: hit rate only checks search. RAGAS also checks the ANSWER:
      faithfulness      = did the answer stick to the retrieved chunks (no made-up claims)?
      answer relevancy  = did the answer actually address the question?
      context recall    = did the retrieved chunks contain what the correct answer needs?"""
    # Imported here so the rest of the project never needs these heavy libraries.
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy

    rows = []
    for q in questions:
        result = rag.ask(q["question"], mode="hybrid_rerank")
        rows.append({
            "user_input": q["question"],
            "response": result["answer"],
            "retrieved_contexts": result["contexts"],
            "reference": q["ground_truth"],
        })
        print(f"  answered {len(rows)}/{len(questions)}")

    # The grader is the SAME OpenAI-compatible LLM from .env; embeddings are the same
    # free local model the pipeline uses, so evaluation costs nothing extra for embeddings.
    grader = LangchainLLMWrapper(ChatOpenAI(
        base_url=rag.LLM_BASE_URL, api_key=rag.LLM_API_KEY, model=rag.LLM_MODEL, temperature=0,
    ))
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=rag.EMBEDDING_MODEL))

    scores = evaluate(
        EvaluationDataset.from_list(rows),
        # strictness=1: ask for one generated question at a time. Some free providers
        # (like Groq) don't support asking for several replies in one call.
        metrics=[Faithfulness(), ResponseRelevancy(strictness=1), LLMContextRecall()],
        llm=grader,
        embeddings=embeddings,
    )
    table = scores.to_pandas()
    return {
        "faithfulness": float(table["faithfulness"].mean()),
        "answer_relevancy": float(table["answer_relevancy"].mean()),
        "context_recall": float(table["context_recall"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-questions", action="store_true", help="write eval_questions.json")
    parser.add_argument("--skip-ragas", action="store_true", help="only compute hit rates")
    args = parser.parse_args()

    if args.make_questions:
        make_questions()
        return

    questions = load_questions()
    print(f"Evaluating on {len(questions)} questions\n")

    results = {"num_questions": len(questions), "hit_rate": {}}
    for mode in MODES:
        results["hit_rate"][mode] = hit_rate(questions, mode)
        print(f"  hit rate  {mode:14s} {results['hit_rate'][mode]:.0%}")

    if not args.skip_ragas:
        print("\nRunning RAGAS (many LLM calls, takes a few minutes)...")
        results["ragas"] = run_ragas(questions)
        for metric, value in results["ragas"].items():
            print(f"  {metric:17s} {value:.2f}")

    os.makedirs("results", exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
