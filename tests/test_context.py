"""Unit tests for AgentContext, ContextRescorer, and probabilistic viseme disambiguation."""

import pytest
from voiceno.context.rescorer import AgentContext, ContextRescorer, TaskContext


def test_agent_context_initialization():
    ctx = AgentContext(
        active_app="file_manager",
        domain_keywords=["convert", "image", "pdf"],
        file_type="image",
    )
    assert ctx.active_app == "file_manager"
    assert "pdf" in ctx.domain_keywords
    assert ctx.file_type == "image"


def test_probabilistic_grammar_scoring():
    rescorer = ContextRescorer()
    # Coherent English command should score much higher than disjointed non-words
    valid_score = rescorer.compute_agent_grammar_score("CONVERT THIS IMAGE TO PDF")
    nonsense_score = rescorer.compute_agent_grammar_score("I WALKED OVER THE LIBYAN SUPER GLOVE")
    assert valid_score > nonsense_score


def test_viseme_disambiguation_park_mall():
    """Tests resolving bilabial ambiguity [p/b]ark at the [m/b]all into 'park at the mall'."""
    rescorer = ContextRescorer()
    hypotheses = [
        {"text": "I WANT TO BARK AT THE BALL", "score": -8.1},  # visual model ranked bark slightly higher
        {"text": "I WANT TO PARK AT THE MALL", "score": -8.3},
    ]
    rescored = rescorer.rescore_hypotheses(hypotheses)
    assert rescored[0]["text"] == "I WANT TO PARK AT THE MALL"
    assert rescored[0]["rescored_score"] > rescored[1]["rescored_score"]


def test_viseme_disambiguation_dog_bark():
    """Tests that contextual subject agreement chooses 'bark' when preceded by 'dog'."""
    rescorer = ContextRescorer()
    hypotheses = [
        {"text": "THE DOG STARTED TO PARK AT THE BALL", "score": -7.9},
        {"text": "THE DOG STARTED TO BARK AT THE BALL", "score": -8.2},
    ]
    rescored = rescorer.rescore_hypotheses(hypotheses)
    assert rescored[0]["text"] == "THE DOG STARTED TO BARK AT THE BALL"


def test_viseme_disambiguation_kite_bite():
    """Tests resolving [k/t/b]ite into 'kite' when preceded by 'buy a'."""
    rescorer = ContextRescorer()
    hypotheses = [
        {"text": "I WANT TO BUY A BITE", "score": -6.5},
        {"text": "I WANT TO BUY A KITE", "score": -6.7},
    ]
    rescored = rescorer.rescore_hypotheses(hypotheses)
    assert rescored[0]["text"] == "I WANT TO BUY A KITE"
