"""Unit tests for AgentContext, ContextRescorer, and AI Agent semantic reconstruction."""

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


def test_agent_grammar_scoring():
    rescorer = ContextRescorer()
    # Canonical agent command should score high on grammar
    cmd_score = rescorer.compute_agent_grammar_score("CONVERT THIS IMAGE TO PDF")
    random_score = rescorer.compute_agent_grammar_score("I WALKED OVER THE LIBYAN SUPER GLOVE")
    assert cmd_score > random_score


def test_agent_semantic_reconstruction():
    rescorer = ContextRescorer()
    hypotheses = [
        {"text": "I WALKED OVER THE LIBYAN SUPER GLOVE", "score": -14.3},
        {"text": "I WALKED OVER THE LIBBY ZOOM LIB", "score": -15.2},
    ]
    ctx = AgentContext(
        active_app="file_manager",
        domain_keywords=["convert", "image", "pdf"],
        file_type="image",
    )
    result = rescorer.reconstruct_agent_command(hypotheses, context=ctx)
    assert result["text"] == "I WANT TO CONVERT THIS IMAGE TO PDF"
    assert result["action"] == "convert"
    assert "HIGH" in result["confidence"]
