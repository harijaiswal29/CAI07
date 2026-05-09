#!/bin/bash
# Run all tests, fastest first.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "=== Guardrails (fast) ==="
$PY tests/test_guardrails.py
echo

echo "=== Preprocessor (fast) ==="
$PY tests/test_preprocessor.py
echo

echo "=== Retriever (medium — loads embedding + cross-encoder) ==="
$PY tests/test_retriever.py
echo

echo "=== End-to-end (slow — runs queries through flan-t5-large) ==="
$PY tests/test_e2e.py
echo

echo "=== All tests passed ==="
