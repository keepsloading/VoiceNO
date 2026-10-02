"""Tests for evaluation metrics."""

import pytest
from voiceno.evaluation.metrics import compute_wer, compute_cer, compute_aggregate_metrics, normalize_text


def test_normalize_text():
    assert normalize_text("Hello, World!") == "HELLO WORLD"
    assert normalize_text("  spaces   and tabs\t") == "SPACES AND TABS"


def test_compute_wer_cer():
    ref = "THE QUICK BROWN FOX"
    hyp = "THE QUICK BROWN DOG"
    wer = compute_wer(ref, hyp)
    cer = compute_cer(ref, hyp)
    assert 0.24 < wer < 0.26
    assert 0.09 < cer < 0.12


def test_identical_strings():
    assert compute_wer("EXACT MATCH", "EXACT MATCH") == 0.0
    assert compute_cer("EXACT MATCH", "EXACT MATCH") == 0.0


def test_aggregate_metrics():
    refs = ["HELLO WORLD", "SILENT SPEECH"]
    hyps = ["HELLO WORLD", "SILENT SPEECH"]
    res = compute_aggregate_metrics(refs, hyps)
    assert res["wer"] == 0.0
    assert res["cer"] == 0.0


if __name__ == "__main__":
    test_normalize_text()
    test_compute_wer_cer()
    test_identical_strings()
    test_aggregate_metrics()
    print("All metric tests passed!")
