"""Personalization engine for VoiceNo! V0.

Implements two stages of adaptation:
- Stage 1: Lightweight visual prompt adaptation
- Stage 2: LoRA (Low-Rank Adaptation) on attention projections (Wq, Wk, Wv)
  freezing base weights and persisting only the tiny personalization delta.
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from voiceno.config import LoRAConfig


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear layer with low-rank decomposition matrices A and B."""

    def __init__(
        self,
        original_linear: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.original_linear = original_linear
        # Ensure original parameters are frozen
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        in_features = original_linear.in_features
        out_features = original_linear.out_features

        dev = original_linear.weight.device
        dtype = original_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(r, in_features, device=dev, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=dev, dtype=dtype))
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Initialize A with Kaiming uniform, B with zeros (ensures zero delta at initialization)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original_linear(x)
        if self.lora_A.device != x.device or self.lora_A.dtype != x.dtype:
            self.lora_A.data = self.lora_A.data.to(device=x.device, dtype=x.dtype)
            self.lora_B.data = self.lora_B.data.to(device=x.device, dtype=x.dtype)
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out


class VisualPromptAdapter(nn.Module):
    """Stage 1: Lightweight learnable prompt tokens prepended to sequence embeddings."""

    def __init__(self, prompt_len: int = 4, embed_dim: int = 768):
        super().__init__()
        self.prompt_len = prompt_len
        self.prompt = nn.Parameter(torch.randn(1, prompt_len, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B = x.size(0)
        prompts = self.prompt.expand(B, -1, -1)
        return torch.cat([prompts, x], dim=1)


class PersonalizationManager:
    """Manages injection, removal, checkpoint saving, and loading of personalization adapters."""

    def __init__(self, config: Optional[LoRAConfig] = None):
        self.config = config or LoRAConfig()
        self.lora_layers: Dict[str, LoRALinear] = {}
        self.prompt_adapter: Optional[VisualPromptAdapter] = None
        self.is_personalized: bool = False
        self.active_mode: str = "Generic"  # 'Generic' or 'Personalized'
        self.target_modules: List[str] = list(self.config.target_modules)
        self.strategy: str = "conformer_lora"

    def apply_lora(
        self,
        model: nn.Module,
        target_module_names: Optional[List[str]] = None,
        strategy: Optional[str] = None,
    ) -> int:
        """Injects LoRA into target linear projections of the model.

        Supports:
        - conformer_lora: Attention projections (Wq, Wk, Wv) across 12 Conformer blocks
        - head_adapter: CTC projection layer (ctc_lo) for instant low-RAM CPU calibration

        Idempotent: Cleanly removes any existing LoRA wrappers first to restore base nn.Linear layers.
        Returns total number of trainable LoRA parameters.
        """
        if strategy is not None:
            self.strategy = strategy
            if strategy == "head_adapter":
                target_module_names = ["ctc_lo"]

        if target_module_names is None:
            target_module_names = self.target_modules or self.config.target_modules

        self.target_modules = list(target_module_names)

        # 1. Restore any previously wrapped LoRA layers back to original Linear layers
        for name, module in model.named_modules():
            for child_name, child in list(module.named_children()):
                if isinstance(child, LoRALinear):
                    setattr(module, child_name, child.original_linear)

        # 2. Freeze entire base model
        for param in model.parameters():
            param.requires_grad = False

        self.lora_layers.clear()
        trainable_params = 0

        # 3. Recursively find matching projection layers and wrap with LoRA
        for name, module in model.named_modules():
            for child_name, child in list(module.named_children()):
                if child_name in target_module_names and isinstance(child, nn.Linear):
                    lora_wrapper = LoRALinear(
                        original_linear=child,
                        r=self.config.r,
                        lora_alpha=self.config.lora_alpha,
                        lora_dropout=self.config.lora_dropout,
                    )
                    lora_wrapper.lora_A.requires_grad = True
                    lora_wrapper.lora_B.requires_grad = True
                    lora_wrapper.to(device=child.weight.device, dtype=child.weight.dtype)
                    setattr(module, child_name, lora_wrapper)
                    layer_key = f"{name}.{child_name}" if name else child_name
                    self.lora_layers[layer_key] = lora_wrapper
                    trainable_params += lora_wrapper.lora_A.numel() + lora_wrapper.lora_B.numel()

        self.is_personalized = True
        self.active_mode = "Personalized"
        return trainable_params

    def remove_lora(self, model: nn.Module) -> None:
        """Removes LoRA adapters and restores original frozen linear layers."""
        for name, module in model.named_modules():
            for child_name, child in list(module.named_children()):
                if isinstance(child, LoRALinear):
                    setattr(module, child_name, child.original_linear)

        self.lora_layers.clear()
        self.is_personalized = False
        self.active_mode = "Generic"

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Returns only the trainable LoRA parameters (lora_A and lora_B)."""
        state_dict = {}
        for key, layer in self.lora_layers.items():
            state_dict[f"{key}.lora_A"] = layer.lora_A.data.clone()
            state_dict[f"{key}.lora_B"] = layer.lora_B.data.clone()
        return state_dict

    def save_profile(
        self,
        user_id: str,
        profile_dir: Optional[Union[str, Path]] = None,
        metadata: Optional[Dict] = None,
    ) -> Path:
        """Saves only the user's personalization delta weights and metadata."""
        if profile_dir is None:
            profile_dir = Path(self.config.profiles_dir) / user_id
        else:
            profile_dir = Path(profile_dir)

        profile_dir.mkdir(parents=True, exist_ok=True)
        weights_path = profile_dir / "lora.pt"
        meta_path = profile_dir / "metadata.json"

        lora_weights = self.get_lora_state_dict()
        torch.save(lora_weights, str(weights_path))

        meta = {
            "user_id": user_id,
            "r": self.config.r,
            "lora_alpha": self.config.lora_alpha,
            "target_modules": self.target_modules,
            "strategy": self.strategy,
            "num_adapted_layers": len(self.lora_layers),
            "num_parameters": sum(t.numel() for t in lora_weights.values()),
        }
        if metadata:
            meta.update(metadata)

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return weights_path

    def load_profile(
        self,
        model: nn.Module,
        user_id: str,
        profile_dir: Optional[Union[str, Path]] = None,
    ) -> Dict:
        """Loads a user's LoRA delta weights into the model."""
        if profile_dir is None:
            profile_dir = Path(self.config.profiles_dir) / user_id
        else:
            profile_dir = Path(profile_dir)

        weights_path = profile_dir / "lora.pt"
        meta_path = profile_dir / "metadata.json"

        if not weights_path.is_file():
            raise FileNotFoundError(f"Personalization weights not found at: {weights_path}")

        metadata = {}
        target_modules = self.config.target_modules
        if meta_path.is_file():
            with open(meta_path) as f:
                metadata = json.load(f)
            target_modules = metadata.get("target_modules", target_modules)
            self.strategy = metadata.get("strategy", "conformer_lora")

        # Inject LoRA layers matching saved target modules
        self.apply_lora(model, target_module_names=target_modules)

        lora_weights = torch.load(str(weights_path), map_location="cpu")
        for key, layer in self.lora_layers.items():
            a_key = f"{key}.lora_A"
            b_key = f"{key}.lora_B"
            if a_key in lora_weights and b_key in lora_weights:
                layer.lora_A.data.copy_(lora_weights[a_key])
                layer.lora_B.data.copy_(lora_weights[b_key])

        self.is_personalized = True
        self.active_mode = "Personalized"

        return metadata

