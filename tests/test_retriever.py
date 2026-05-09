"""Unit tests for retrieval and cross-encoder reranking in rag_implementation.py.

Loads the embedding model + cross-encoder once. ~30 s on first run (model
downloads), ~5 s on cached runs. Does NOT load the LLM — we monkeypatch the
LLM init to keep memory low and runs fast.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress LLM load — we don't need it for retrieval-only tests.
# Patch BEFORE importing RAGSystem.
from transformers import T5ForConditionalGeneration, AutoTokenizer
import rag_implementation as _rag_mod

class _StubLLM:
    def generate(self, *a, **kw):
        raise RuntimeError("LLM stub: generate() should not be called in retrieval tests")

class _StubTokenizer:
    def __call__(self, *a, **kw):
        raise RuntimeError("LLM stub: tokenizer should not be called in retrieval tests")
    def decode(self, *a, **kw):
        return ""

_orig_t5 = T5ForConditionalGeneration.from_pretrained
_orig_tok = AutoTokenizer.from_pretrained
T5ForConditionalGeneration.from_pretrained = lambda *a, **kw: _StubLLM()
AutoTokenizer.from_pretrained = lambda *a, **kw: _StubTokenizer()

from rag_implementation import RAGSystem, AdvancedRetrieval  # noqa: E402
from tests.sample_data import REVENUE_TRAP_CORPUS  # noqa: E402

# Restore originals so anything after this still works
T5ForConditionalGeneration.from_pretrained = _orig_t5
AutoTokenizer.from_pretrained = _orig_tok


def _check(label, condition, detail=""):
    if not condition:
        print(f"  FAIL: {label}{(' — ' + detail) if detail else ''}")
        sys.exit(1)
    print(f"  OK:   {label}")


# ---- Fixtures shared across tests (built once) ----
print("Building test RAG system (loads embedding model)...")
# Use unique paths so we don't clobber the real index on disk.
_test_index = "/tmp/finrag_test_index"
_test_chunks = "/tmp/finrag_test_chunks.pkl"
for p in (_test_index, _test_chunks):
    if os.path.exists(p):
        os.remove(p)
RAG = RAGSystem(index_path=_test_index, chunks_path=_test_chunks)
RAG.build_index(REVENUE_TRAP_CORPUS)


def test_embedding_scores_in_range():
    print("test_embedding_scores_in_range")
    results = RAG.retrieve("total revenue 2023", k=5)
    _check("got 5 results", len(results) == 5)
    for r in results:
        _check(f"score {r['score']:.3f} in [0, 1]",
               0.0 <= r["score"] <= 1.0,
               detail=f"chunk={r['chunk'][:60]!r}")


def test_reranker_demotes_tax_provision_trap():
    print("test_reranker_demotes_tax_provision_trap")
    # Historical bug: ms-marco-MiniLM ranked the "Provision for income taxes"
    # chunk #1 at confidence 1.0 on revenue queries (shared surface vocab:
    # "income", year). BGE reranker should demote this chunk below #1.
    query = "What was the total revenue for 2023?"
    candidates = RAG.retrieve(query, k=5)
    reranked = RAG._rerank_with_cross_encoder(query, candidates, top_k=5)
    top_chunk = reranked[0]["chunk"]
    _check("tax-provision chunk is NOT top-1 after rerank (the original bug)",
           "Provision for income taxes" not in top_chunk,
           detail=f"top_chunk={top_chunk!r}")
    # Find the tax chunk's rank — it should be in the bottom half.
    tax_rank = next(
        (i for i, r in enumerate(reranked, 1)
         if "Provision for income taxes" in r["chunk"]),
        None,
    )
    if tax_rank is not None:
        _check(f"tax chunk demoted to rank >= 3 (got rank {tax_rank})",
               tax_rank >= 3)


def test_reranker_scores_sigmoid_calibrated():
    print("test_reranker_scores_sigmoid_calibrated")
    candidates = RAG.retrieve("total revenue 2023", k=5)
    reranked = RAG._rerank_with_cross_encoder("total revenue 2023", candidates, top_k=5)
    for r in reranked:
        _check(f"reranked score {r['score']:.3f} in [0, 1]",
               0.0 <= r["score"] <= 1.0)
    # Top score should be < 0.999 — a fully-saturated 1.0 is the old *2.0
    # logit-sharpening bug returning. Allow up to 0.999 for genuinely strong matches.
    _check("top score not artificially saturating (< 0.999)",
           reranked[0]["score"] < 0.999,
           detail=f"top score = {reranked[0]['score']:.6f}")


def test_bm25_returns_nonzero_for_keyword_match():
    print("test_bm25_returns_nonzero_for_keyword_match")
    adv = AdvancedRetrieval()
    adv.initialize_bm25(REVENUE_TRAP_CORPUS)
    results = adv.retrieve_with_bm25("provision income taxes", REVENUE_TRAP_CORPUS, top_k=3)
    _check("got results", len(results) > 0)
    _check("top BM25 score > 0", results[0]["score"] > 0.0)
    _check("top result mentions 'income taxes'",
           "income taxes" in results[0]["chunk"].lower(),
           detail=f"top={results[0]['chunk']!r}")


def main():
    test_embedding_scores_in_range()
    test_reranker_demotes_tax_provision_trap()
    test_reranker_scores_sigmoid_calibrated()
    test_bm25_returns_nonzero_for_keyword_match()
    print("\nAll retriever tests passed.")


if __name__ == "__main__":
    main()
