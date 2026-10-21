"""Tests for LoRA personalization and parameter management."""

import pytest
import torch
import torch.nn as nn

from memact.models.personalization import LoRALinear, PersonalizationManager, VisualPromptAdapter
from memact.config import LoRAConfig


class DummyAttentionBlock(nn.Module):
    """Simulates an attention block with linear_q, linear_k, linear_v."""
    def __init__(self, dim=64):
        super().__init__()
        self.linear_q = nn.Linear(dim, dim)
        self.linear_k = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, dim)
        self.linear_out = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        return self.linear_out(q + k + v)


def test_lora_initial_delta_is_zero():
    """LoRA output must match original linear layer exactly at initialization."""
    orig = nn.Linear(32, 32)
    lora = LoRALinear(orig, r=4, lora_alpha=8)

    x = torch.randn(2, 10, 32)
    with torch.no_grad():
        out_orig = orig(x)
        out_lora = lora(x)

    torch.testing.assert_close(out_orig, out_lora)


def test_personalization_manager_injection_and_removal():
    """Verify that manager injects into Wq, Wk, Wv and freezes base parameters."""
    model = nn.Sequential(
        DummyAttentionBlock(64),
        DummyAttentionBlock(64),
    )

    manager = PersonalizationManager()
    trainable_params = manager.apply_lora(model)

    # 2 blocks * 3 projections (q, k, v) = 6 LoRA layers
    assert len(manager.lora_layers) == 6
    assert trainable_params > 0

    # Base model parameters must be frozen
    for name, param in model.named_parameters():
        if "lora" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False

    # Test removal
    manager.remove_lora(model)
    assert len(manager.lora_layers) == 0
    assert manager.is_personalized is False


def test_save_and_load_profile(tmp_path):
    """Verify that saving and loading profiles restores LoRA weights."""
    model = DummyAttentionBlock(32)
    manager = PersonalizationManager()
    manager.apply_lora(model)

    # Simulate some training weight delta
    with torch.no_grad():
        for layer in manager.lora_layers.values():
            layer.lora_B.fill_(0.5)

    profile_dir = tmp_path / "test_user"
    weights_path = manager.save_profile("test_user", profile_dir=profile_dir)
    assert weights_path.is_file()

    # Create new model and load profile
    new_model = DummyAttentionBlock(32)
    new_manager = PersonalizationManager()
    new_manager.load_profile(new_model, "test_user", profile_dir=profile_dir)

    for layer in new_manager.lora_layers.values():
        torch.testing.assert_close(layer.lora_B, torch.full_like(layer.lora_B, 0.5))


def test_profile_restores_saved_rank_and_scaling(tmp_path):
    manager = PersonalizationManager(LoRAConfig(r=4, lora_alpha=12))
    manager.apply_lora(DummyAttentionBlock(8))
    manager.save_profile("speaker", tmp_path)
    loaded = PersonalizationManager()
    loaded.load_profile(DummyAttentionBlock(8), "speaker", tmp_path)
    assert loaded.active_user_id == "speaker"
    assert all(layer.r == 4 and layer.scaling == 3 for layer in loaded.lora_layers.values())


def test_incomplete_profile_is_rejected_and_adapters_removed(tmp_path):
    manager = PersonalizationManager()
    manager.apply_lora(DummyAttentionBlock(8))
    path = manager.save_profile("speaker", tmp_path)
    weights = torch.load(path, weights_only=True)
    del weights[next(iter(weights))]
    torch.save(weights, path)
    fresh_model = DummyAttentionBlock(8)
    fresh = PersonalizationManager()
    with pytest.raises(ValueError, match="do not match"):
        fresh.load_profile(fresh_model, "speaker", tmp_path)
    assert not fresh.is_personalized
    assert fresh.active_user_id is None
    assert not any(isinstance(layer, LoRALinear) for layer in fresh_model.modules())


if __name__ == "__main__":
    test_lora_initial_delta_is_zero()
    test_personalization_manager_injection_and_removal()
    print("All personalization tests passed!")
