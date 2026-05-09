"""Unit tests for the sentence-aware chunker in financial_preprocessor.py.

No model loading. Should run in < 1 second.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from financial_preprocessor import FinancialDataPreprocessor
from tests.sample_data import SAMPLE_10K_TEXT


def _check(label, condition, detail=""):
    if not condition:
        print(f"  FAIL: {label}{(' — ' + detail) if detail else ''}")
        sys.exit(1)
    print(f"  OK:   {label}")


def test_decimals_preserved():
    print("test_decimals_preserved")
    p = FinancialDataPreprocessor(chunk_size=500, chunk_overlap=80)
    chunks = p.create_chunks(SAMPLE_10K_TEXT)
    joined = " ||| ".join(chunks)
    for needle in ["$309.1 billion", "28.2 billion", "$280.9 billion",
                   "$84.3 billion", "$74.8 billion", "13.9 percent",
                   "15.3 percent", "110.9 billion", "$73.8 billion",
                   "$45.4 billion", "$32.3 billion", "$69.5 billion"]:
        # Each decimal phrase must appear in at least one chunk as one substring
        # — i.e. it must NOT have been split mid-number across two chunks.
        in_chunk = any(needle in c for c in chunks)
        _check(f"'{needle}' kept in one chunk", in_chunk,
               detail=f"chunks={chunks!r}")


def test_abbreviations_preserved():
    print("test_abbreviations_preserved")
    p = FinancialDataPreprocessor(chunk_size=500, chunk_overlap=80)
    chunks = p.create_chunks(SAMPLE_10K_TEXT)
    # If "Inc." had been treated as a sentence terminator, "reported" would
    # land in a different chunk than "Alphabet Inc."
    _check("'Alphabet Inc. reported' stays together",
           any("Alphabet Inc. reported" in c for c in chunks))
    _check("'U.S. operations' stays together",
           any("U.S. operations" in c for c in chunks))
    _check("'e.g. lower' stays together",
           any("e.g. lower" in c for c in chunks))


def test_overlap_actually_applied():
    print("test_overlap_actually_applied")
    # Force multiple chunks by using a small chunk_size.
    p = FinancialDataPreprocessor(chunk_size=200, chunk_overlap=80)
    chunks = p.create_chunks(SAMPLE_10K_TEXT)
    _check("produced multiple chunks for overlap test", len(chunks) >= 2,
           detail=f"got {len(chunks)} chunks")

    # The overlap mechanism prepends trailing sentences of chunk[i] to chunk[i+1].
    # So the first ~40 chars of chunk[i+1] should appear somewhere in chunk[i].
    # (Sometimes chunk[i] is only one big sentence; in that case head matches start of chunk[i].)
    for i in range(len(chunks) - 1):
        head_next = chunks[i + 1][:40]
        shared = head_next in chunks[i]
        _check(f"chunks[{i+1}] starts with content also in chunks[{i}] (overlap real)",
               shared, detail=f"head_next={head_next!r}")


def test_chunk_size_respected():
    print("test_chunk_size_respected")
    chunk_size = 200
    p = FinancialDataPreprocessor(chunk_size=chunk_size, chunk_overlap=50)
    chunks = p.create_chunks(SAMPLE_10K_TEXT)
    # Allow some slack — a single sentence may push past chunk_size since we
    # only flush *before* adding the next sentence.
    max_allowed = int(chunk_size * 1.5)
    for i, c in enumerate(chunks):
        _check(f"chunks[{i}] within size budget ({len(c)} <= {max_allowed})",
               len(c) <= max_allowed)


def test_empty_input():
    print("test_empty_input")
    p = FinancialDataPreprocessor(chunk_size=200, chunk_overlap=50)
    chunks = p.create_chunks("")
    _check("empty string returns empty list", chunks == [])


def test_single_sentence():
    print("test_single_sentence")
    p = FinancialDataPreprocessor(chunk_size=200, chunk_overlap=50)
    chunks = p.create_chunks("Just one short sentence.")
    _check("single sentence yields one chunk", len(chunks) == 1)
    _check("content preserved",
           "Just one short sentence." in chunks[0])


def main():
    test_decimals_preserved()
    test_abbreviations_preserved()
    test_overlap_actually_applied()
    test_chunk_size_respected()
    test_empty_input()
    test_single_sentence()
    print("\nAll preprocessor tests passed.")


if __name__ == "__main__":
    main()
