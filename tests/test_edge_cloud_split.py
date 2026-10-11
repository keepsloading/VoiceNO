"""Unit tests for VoiceNO Edge-Cloud Split processing and Confidential Cloud API."""

import numpy as np
import torch
import pytest

from voiceno.client.mouth_extractor import EdgeMouthExtractor
from voiceno.domains import DomainRegistry, get_active_domain, set_active_domain
from voiceno.server.api import CloudInferenceEngine


def test_edge_mouth_extractor_serialization():
    """Tests that local mouth tensor serializes to < 300 KB payload and deserializes cleanly."""
    extractor = EdgeMouthExtractor()

    # 35 frames (~1.4s PTT clip) of 88x88 mouth patches
    synthetic_mouth = torch.randn(35, 1, 88, 88).float()
    payload_bytes = extractor.serialize_payload(synthetic_mouth, compress=True)

    # Verify payload size is well under 300 KB
    size_kb = len(payload_bytes) / 1024
    assert size_kb < 300.0, f"Payload size {size_kb:.2f} KB exceeded 300 KB threshold"

    # Verify deserialization preserves exact dimensions
    recovered = EdgeMouthExtractor.deserialize_payload(payload_bytes)
    assert recovered.shape == (35, 1, 88, 88)
    assert recovered.dtype == torch.float32
    # Verify quantization fidelity
    assert torch.max(torch.abs(synthetic_mouth - recovered)) < 0.1




def test_all_domains_registration_and_activation():
    """Tests that all 4 roadmap domains are registered and switchable."""
    domains = DomainRegistry.list_domains()
    assert "ai_prompts" in domains
    assert "coding" in domains
    assert "email" in domains
    assert "blogging" in domains

    # Test activating coding domain
    set_active_domain("coding")
    assert get_active_domain().name == "coding"
    coding_bias = get_active_domain().compute_domain_bias("GIT COMMIT AND PUSH")
    assert coding_bias > 5.0

    # Test activating email domain
    set_active_domain("email")
    assert get_active_domain().name == "email"
    email_bias = get_active_domain().compute_domain_bias("PLEASE LET ME KNOW ATTACHED IS THE FILE")
    assert email_bias > 5.0

    # Test activating blogging domain
    set_active_domain("blogging")
    assert get_active_domain().name == "blogging"
    blog_bias = get_active_domain().compute_domain_bias("IN THIS ARTICLE WE EXPLORE")
    assert blog_bias > 5.0

    # Reset to default ai_prompts
    set_active_domain("ai_prompts")
    assert get_active_domain().name == "ai_prompts"


def test_cloud_inference_engine_structure():
    """Tests CloudInferenceEngine response structure and ephemeral execution."""
    class DummyModel:
        def __init__(self):
            self.device = "cpu"
        def transcribe_tensor(self, tensor, **kwargs):
            return {
                "hypotheses": [
                    {"text": "CONVERT THIS IMAGE TO PDF", "score": -4.0},
                    {"text": "LIKE IT'S IN MY SCHOOL", "score": -4.2},
                ]
            }

    engine = CloudInferenceEngine(model=DummyModel())
    extractor = EdgeMouthExtractor()
    synthetic_mouth = torch.randn(25, 1, 88, 88).float()
    payload = extractor.serialize_payload(synthetic_mouth)

    result = engine.process_mouth_payload(payload, domain_name="ai_prompts")
    assert result["text"] == "CONVERT THIS IMAGE TO PDF"
    assert result["domain"] == "ai_prompts"
    assert "latency_ms" in result
    assert "privacy" in result
