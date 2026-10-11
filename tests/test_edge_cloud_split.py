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




def test_ai_prompts_domain_registration():
    """Tests that the AI Prompts domain is registered and active."""
    domains = DomainRegistry.list_domains()
    assert "ai_prompts" in domains
    assert get_active_domain().name == "ai_prompts"

    # Test agent prompt bias
    prompt_bias = get_active_domain().compute_domain_bias("CONVERT THIS IMAGE TO PDF")
    assert prompt_bias > 5.0



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
