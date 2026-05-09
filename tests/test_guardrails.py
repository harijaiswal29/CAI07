"""Unit tests for input/output guardrails in app.py.

No model loading — pure string and threshold logic. Should run in < 1 second.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import validate_query, format_confidence


def _check(label, condition):
    if not condition:
        print(f"  FAIL: {label}")
        sys.exit(1)
    print(f"  OK:   {label}")


def test_empty_query_rejected():
    print("test_empty_query_rejected")
    _check("empty string rejected", validate_query("")["valid"] is False)
    _check("whitespace-only rejected", validate_query("   ")["valid"] is False)


def test_too_short_rejected():
    print("test_too_short_rejected")
    _check("'hi' rejected", validate_query("hi")["valid"] is False)
    _check("4-char query rejected", validate_query("abcd")["valid"] is False)


def test_irrelevant_query_rejected():
    print("test_irrelevant_query_rejected")
    _check(
        "weather query rejected (no financial keyword)",
        validate_query("What is the weather today in Paris?")["valid"] is False,
    )
    _check(
        "cake recipe rejected",
        validate_query("How do I bake a chocolate cake?")["valid"] is False,
    )


def test_relevant_query_accepted():
    print("test_relevant_query_accepted")
    examples = [
        "What was the total revenue for 2023?",
        "How much did operating expenses increase from 2022 to 2023?",
        "What are the main factors affecting profitability?",
        "How has the cash position changed year-over-year?",
    ]
    for q in examples:
        _check(f"accepted: {q!r}", validate_query(q)["valid"] is True)


def test_format_confidence_thresholds():
    print("test_format_confidence_thresholds")
    cls_high, label_high = format_confidence(0.8)
    cls_med, label_med = format_confidence(0.5)
    cls_low, label_low = format_confidence(0.2)
    _check("0.8 -> High", "high" in cls_high.lower() and "High" in label_high)
    _check("0.5 -> Medium", "medium" in cls_med.lower() and "Medium" in label_med)
    _check("0.2 -> Low", "low" in cls_low.lower() and "Low" in label_low)
    # Boundary checks
    _check("0.7 -> High (boundary)", "high" in format_confidence(0.7)[0].lower())
    _check("0.4 -> Medium (boundary)", "medium" in format_confidence(0.4)[0].lower())
    _check("0.39 -> Low", "low" in format_confidence(0.39)[0].lower())


def main():
    test_empty_query_rejected()
    test_too_short_rejected()
    test_irrelevant_query_rejected()
    test_relevant_query_accepted()
    test_format_confidence_thresholds()
    print("\nAll guardrail tests passed.")


if __name__ == "__main__":
    main()
