"""Personalization trainer for VoiceNo! V0.

Trains user-specific LoRA parameters on silently articulated calibration videos
while keeping the foundation Auto-AVSR model strictly frozen.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch
import torch.optim as optim

from voiceno.config import CalibrationConfig, LoRAConfig
from voiceno.models.auto_avsr import AutoAVSRModel
from voiceno.models.personalization import PersonalizationManager
from voiceno.preprocessing.normalize import VideoPreprocessor


class PersonalizationTrainer:
    """Orchestrates local LoRA calibration training for a user profile."""

    def __init__(
        self,
        model_wrapper: AutoAVSRModel,
        personalization_mgr: Optional[PersonalizationManager] = None,
        calib_config: Optional[CalibrationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ):
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        self.tokenizer = model_wrapper.text_transform
        self.calib_config = calib_config or CalibrationConfig()
        self.lora_config = lora_config or LoRAConfig()
        self.personalization_mgr = personalization_mgr or PersonalizationManager(self.lora_config)
        self.preprocessor = VideoPreprocessor()

    def train_on_utterances(
        self,
        user_id: str,
        utterance_items: List[Dict],
        max_duration_seconds: Optional[float] = None,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        strategy: str = "fast_cpu",
    ) -> Dict:
        """Runs LoRA adaptation on a list of calibration utterance dicts.

        Args:
            user_id: User profile identifier (e.g. 'default', 'user_001')
            utterance_items: List of dicts containing 'video_path' and 'prompt'
            max_duration_seconds: Optional limit to simulate 1m, 5m calibration experiments
            epochs: Number of optimization epochs
            lr: Learning rate
            strategy: 'fast_cpu' (instant feature-cached CTC head adapter, <50MB RAM)
                      or 'conformer_lora' (full 12-layer attention LoRA for GPU/Colab)
        """
        if self.model is None or self.model_wrapper.model is None:
            self.model_wrapper.load_checkpoint()
            self.model = self.model_wrapper.model

        epochs = epochs or self.calib_config.epochs
        lr = lr or (1e-3 if strategy == "fast_cpu" else self.calib_config.learning_rate)

        # Filter utterances by cumulative duration if requested
        selected_items = []
        cumulative_sec = 0.0
        for item in utterance_items:
            dur = item.get("duration_seconds", 3.0)
            if max_duration_seconds is not None and (cumulative_sec + dur) > max_duration_seconds:
                break
            selected_items.append(item)
            cumulative_sec += dur

        if not selected_items:
            selected_items = utterance_items[:1]
            cumulative_sec = selected_items[0].get("duration_seconds", 3.0)

        # Preprocess calibration samples into tensors & token targets
        processed_data = []
        for item in selected_items:
            v_path = item["video_path"]
            prompt = item["prompt"]
            target_ids = self.tokenizer.tokenize(prompt).to(self.model_wrapper.device)

            import cv2
            cap = cv2.VideoCapture(str(v_path))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

            if len(frames) < 5:
                continue

            try:
                tensor, _ = self.preprocessor.process(frames)
                tensor = tensor.to(self.model_wrapper.device)
                processed_data.append((tensor, target_ids, item))
            except Exception as e:
                print(f"Skipping utterance {v_path} due to preprocessing error: {e}")
                continue

        if not processed_data:
            raise RuntimeError("No valid calibration samples could be preprocessed")

        history = []
        start_time = time.time()

        if strategy == "fast_cpu":
            # 1. Feature Caching in torch.no_grad()
            # Runs frontend + conformer encoder once without retaining backwards graphs.
            # Consumes < 50MB RAM and runs in ~0.5s per utterance.
            self.model.eval()
            cached_features = []
            with torch.no_grad():
                for tensor, target_ids, item in processed_data:
                    # tensor: (T, 1, 88, 88) -> (1, T, 1, 88, 88)
                    x = self.model.frontend(tensor.unsqueeze(0))
                    x = self.model.proj_encoder(x)
                    enc_feat, _ = self.model.encoder(x, None)
                    cached_features.append((enc_feat.squeeze(0), target_ids, item))

            # 2. Inject LoRA on CTC projection layer (ctc_lo)
            trainable_params = self.personalization_mgr.apply_lora(
                self.model,
                target_module_names=["ctc_lo"],
                strategy="head_adapter",
            )

            trainable_vars = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(
                trainable_vars,
                lr=lr,
                weight_decay=self.calib_config.weight_decay,
            )
            ctc_loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=True)

            self.model.ctc.train()
            for epoch in range(1, epochs + 1):
                epoch_loss = 0.0
                for enc_feat, target_ids, item in cached_features:
                    optimizer.zero_grad()
                    # enc_feat: (T, 768)
                    logits = self.model.ctc.ctc_lo(enc_feat)  # (T, odim)
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).unsqueeze(1)  # (T, 1, odim)
                    in_len = torch.tensor([enc_feat.size(0)], device=self.model_wrapper.device)
                    tgt_len = torch.tensor([target_ids.size(0)], device=self.model_wrapper.device)

                    loss = ctc_loss_fn(log_probs, target_ids.unsqueeze(0), in_len, tgt_len)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_vars, max_norm=5.0)
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(cached_features)
                history.append({"epoch": epoch, "loss": round(avg_loss, 4)})

        else:
            # Full Conformer Attention LoRA (Suitable for GPU / Cloud execution)
            trainable_params = self.personalization_mgr.apply_lora(
                self.model,
                strategy="conformer_lora",
            )
            trainable_vars = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(
                trainable_vars,
                lr=lr,
                weight_decay=self.calib_config.weight_decay,
            )

            self.model.train()
            for epoch in range(1, epochs + 1):
                epoch_loss = 0.0
                for tensor, target_ids, item in processed_data:
                    optimizer.zero_grad()
                    inputs = tensor.unsqueeze(0)
                    input_lengths = torch.tensor([tensor.size(0)], device=self.model_wrapper.device)
                    targets = target_ids.unsqueeze(0)

                    loss, loss_ctc, loss_att, acc = self.model(inputs, input_lengths, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_vars, max_norm=5.0)
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(processed_data)
                history.append({"epoch": epoch, "loss": round(avg_loss, 4)})

        training_time = time.time() - start_time
        self.model.eval()

        # Save profile
        profile_meta = {
            "calibration_minutes": round(cumulative_sec / 60.0, 2),
            "num_calibration_utterances": len(processed_data),
            "training_time_seconds": round(training_time, 2),
            "final_loss": history[-1]["loss"] if history else 0.0,
            "strategy": strategy,
        }
        profile_path = self.personalization_mgr.save_profile(
            user_id=user_id,
            metadata=profile_meta,
        )

        return {
            "user_id": user_id,
            "profile_path": str(profile_path),
            "strategy": strategy,
            "trainable_parameters": trainable_params,
            "calibration_minutes": round(cumulative_sec / 60.0, 2),
            "training_seconds": round(training_time, 2),
            "history": history,
        }

