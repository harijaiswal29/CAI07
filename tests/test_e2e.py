"""End-to-end query battery for the Financial RAG system.

Loads the full RAG stack (embedding + cross-encoder + flan-t5-large) and runs
a curated battery of queries through the pipeline, printing a results table.

Requires faiss_index + chunks.pkl to exist (built by uploading PDFs in the
Streamlit UI). Slow: ~10-30 s per query × 7 queries on CPU.

Soft assertions only — prints WARN for unexpected confidence levels but
exits 0 unless a query crashes or returns an empty response.
"""
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INDEX_PATH = "faiss_index"
CHUNKS_PATH = "chunks.pkl"

QUERIES = {
    "high_conf": [
        "What was the total revenue for 2023?",
        "How much did operating expenses increase from 2022 to 2023?",
        "What was the cash position at year-end 2023?",
    ],
    "low_conf": [
        "What is the company's strategy for quantum computing?",
        "How many employees worked from home in 2023?",
    ],
    "irrelevant": [
        "What is the capital of France?",
        "How do I bake a chocolate cake?",
    ],
}

# Soft thresholds — warnings only, not failures.
HIGH_CONF_MIN = 0.4    # high-conf queries should at least clear medium
NOT_HIGH_MAX = 0.7     # low-conf and irrelevant queries shouldn't claim high
PEAK_RSS_MAX_GB = 9.0  # OOM regression sanity check (T5-large + BGE + embeddings ~ 8 GB peak)


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def truncate(s, n=45):
    return s if len(s) <= n else s[: n - 1] + "…"


def main():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH)):
        print(f"ERROR: {INDEX_PATH} or {CHUNKS_PATH} not found.")
        print("Build the index first by uploading PDFs in the Streamlit UI:")
        print("  .venv/bin/streamlit run app.py")
        sys.exit(1)

    print("Loading RAG system (embedding + cross-encoder + flan-t5-large)...")
    print("First run: ~30-60 s. After that: cached.")
    from rag_implementation import RAGSystem

    rag = RAGSystem(
        index_path=INDEX_PATH,
        chunks_path=CHUNKS_PATH,
        enable_advanced_retrieval=True,
    )
    print("RAG system loaded.\n")

    rows = []
    warnings = []
    failures = []

    for category, qs in QUERIES.items():
        for q in qs:
            t0 = time.time()
            try:
                result = rag.query(q)
            except Exception as e:
                failures.append(f"[{category}] {q!r} raised {type(e).__name__}: {e}")
                continue
            elapsed = time.time() - t0

            response = result.get("response", "")
            conf = float(result.get("confidence_score", 0.0))
            chunks = result.get("retrieved_chunks", [])
            top_score = float(chunks[0]["score"]) if chunks else 0.0

            if not response.strip():
                failures.append(f"[{category}] {q!r} returned empty response")
                continue

            rows.append({
                "category": category,
                "query": q,
                "conf": conf,
                "top_src": top_score,
                "elapsed": elapsed,
                "response": response,
            })

            # Soft checks
            if category == "high_conf" and conf < HIGH_CONF_MIN:
                warnings.append(
                    f"[high_conf] confidence {conf:.2f} below {HIGH_CONF_MIN} for {q!r}"
                )
            if category in ("low_conf", "irrelevant") and conf >= NOT_HIGH_MAX:
                warnings.append(
                    f"[{category}] confidence {conf:.2f} above {NOT_HIGH_MAX} for {q!r} "
                    f"(system over-confident on a query it shouldn't know)"
                )

    # ---- Print results table ----
    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    print(f"{'Category':<11} {'Query':<46} {'Conf':>5} {'TopSrc':>7} {'Elapsed':>9}")
    print("-" * 100)
    for r in rows:
        print(
            f"{r['category']:<11} {truncate(r['query']):<46} "
            f"{r['conf']:>5.2f} {r['top_src']:>7.3f} {r['elapsed']:>7.1f}s"
        )

    # ---- Print full responses for inspection ----
    print("\n" + "=" * 100)
    print("RESPONSES")
    print("=" * 100)
    for r in rows:
        print(f"\n[{r['category']}] Q: {r['query']}")
        print(f"  Conf={r['conf']:.2f}, TopSrc={r['top_src']:.3f}, "
              f"Elapsed={r['elapsed']:.1f}s")
        print(f"  A: {r['response']}")

    # ---- Memory check ----
    peak_gb = peak_rss_gb()
    print(f"\nPeak RSS during run: {peak_gb:.2f} GB (threshold: {PEAK_RSS_MAX_GB:.1f} GB)")
    if peak_gb > PEAK_RSS_MAX_GB:
        warnings.append(
            f"Peak RSS {peak_gb:.2f} GB exceeded {PEAK_RSS_MAX_GB:.1f} GB — "
            f"OOM regression risk"
        )

    # ---- Summary ----
    print("\n" + "=" * 100)
    print(f"SUMMARY: {len(rows)} queries completed, "
          f"{len(warnings)} warnings, {len(failures)} failures")
    print("=" * 100)
    for w in warnings:
        print(f"  WARN: {w}")
    for f in failures:
        print(f"  FAIL: {f}")

    if failures:
        sys.exit(1)
    print("\nAll E2E queries completed successfully.")


if __name__ == "__main__":
    main()
