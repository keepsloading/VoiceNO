"""Configuration settings for VoiceNo! V0.

Centralizes configuration for model paths, preprocessing,
calibration, LoRA personalization, and evaluation.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Base repository root directory
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PreprocessingConfig:
    """Configuration for face detection and mouth ROI extraction."""
    # Target visual representation: 'mouth', 'mouth_jaw', 'full_face'
    representation: str = "mouth"
    # Dimensions for mouth ROI patch
    crop_width: int = 96
    crop_height: int = 96
    # Model input crop dimension (CenterCrop)
    model_input_size: int = 88
    # Video frame rate
    target_fps: float = 25.0
    # Window margin for temporal landmark smoothing
    window_margin: int = 12
    # Normalization constants matching Auto-AVSR
    norm_mean: float = 0.421
    norm_std: float = 0.165
    # Reference mean face landmarks path
    mean_face_path: Path = field(
        default_factory=lambda: REPO_ROOT / "voiceno" / "preprocessing" / "20words_mean_face.npy"
    )


@dataclass
class ModelConfig:
    """Configuration for Auto-AVSR visual speech recognition backbone."""
    modality: str = "video"
    checkpoint_path: Path = field(
        default_factory=lambda: REPO_ROOT / "checkpoints" / "vsr_trlrs2lrs3vox2avsp_base.pth"
    )
    sp_model_path: Path = field(
        default_factory=lambda: REPO_ROOT / "voiceno" / "models" / "spm" / "unigram" / "unigram5000.model"
    )
    dict_path: Path = field(
        default_factory=lambda: REPO_ROOT / "voiceno" / "models" / "spm" / "unigram" / "unigram5000_units.txt"
    )
    # CTC weight in hybrid CTC/Attention architecture
    ctc_weight: float = 0.1
    # Beam search decoder beam size (can use 10 for fast CPU inference or 40 for max accuracy)
    beam_size: int = 10
    penalty: float = 0.0
    lm_weight: float = 0.0
    device: str = "cpu"


@dataclass
class LoRAConfig:
    """Configuration for LoRA personalization."""
    # Rank of low-rank matrices
    r: int = 8
    # Scaling factor
    lora_alpha: int = 16
    # Dropout probability for LoRA layers
    lora_dropout: float = 0.05
    # Target modules in Conformer / Transformer attention
    target_modules: List[str] = field(
        default_factory=lambda: ["linear_q", "linear_k", "linear_v"]
    )
    # Default directory to save user profiles
    profiles_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "user_profiles"
    )


@dataclass
class CalibrationConfig:
    """Configuration for user calibration and personalization training."""
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    epochs: int = 15
    batch_size: int = 1
    # Target calibration durations in minutes for experiments
    duration_tiers_minutes: List[float] = field(
        default_factory=lambda: [0.0, 1.0, 5.0, 15.0, 30.0, 45.0]
    )
    # Storage directories for recorded calibration utterances
    calibration_data_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "calibration_data"
    )


@dataclass
class VoiceNoConfig:
    """Top-level VoiceNo configuration."""
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
