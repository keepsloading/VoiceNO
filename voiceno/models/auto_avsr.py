"""Auto-AVSR visual speech recognition model wrapper.

Provides clean model loading, checkpoint handling, and transcription pipeline
with support for both beam search and fast CTC greedy decoding.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

# Ensure espnet package within voiceno/models is discoverable
_MODELS_DIR = Path(__file__).resolve().parent
if str(_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODELS_DIR))

from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
import sentencepiece

from voiceno.config import ModelConfig
from voiceno.preprocessing.normalize import VideoPreprocessor


class TextTransform:
    """SentencePiece tokenization and post-processing for Auto-AVSR."""

    def __init__(self, sp_model_path: Union[str, Path], dict_path: Union[str, Path]):
        self.spm = sentencepiece.SentencePieceProcessor(model_file=str(sp_model_path))
        units = open(str(dict_path), encoding="utf8").read().splitlines()
        self.hashmap = {unit.split()[0]: unit.split()[-1] for unit in units}
        # 0 is used for <blank> in CTC
        self.token_list = ["<blank>"] + list(self.hashmap.keys()) + ["<eos>"]
        self.ignore_id = -1

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text to tensor of token IDs."""
        tokens = self.spm.EncodeAsPieces(text.upper())
        token_ids = [self.hashmap.get(token, self.hashmap["<unk>"]) for token in tokens]
        return torch.tensor(list(map(int, token_ids)), dtype=torch.long)

    def post_process(self, token_ids: Union[torch.Tensor, List[int]]) -> str:
        """Decode token IDs back to plain text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().numpy().tolist()
        token_ids = [idx for idx in token_ids if idx != self.ignore_id and idx < len(self.token_list)]
        tokens = [self.token_list[idx] for idx in token_ids]
        text = "".join(tokens).replace("<space>", " ").replace("\u2581", " ").replace("<eos>", "").strip()
        return text


def get_beam_search_decoder(
    model: E2E,
    token_list: List[str],
    beam_size: int = 10,
    ctc_weight: float = 0.1,
    penalty: float = 0.0,
    lm_weight: float = 0.0,
) -> BatchBeamSearch:
    """Builds BatchBeamSearch decoder matching Auto-AVSR configuration."""
    sos = model.odim - 1
    eos = model.odim - 1
    scorers = model.scorers()
    scorers["lm"] = None
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": lm_weight,
        "length_bonus": penalty,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=sos,
        eos=eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )


class AutoAVSRModel:
    """Encapsulates the Auto-AVSR pretrained visual speech recognition model."""

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() and self.config.device != "cpu" else "cpu")
        self.text_transform = TextTransform(
            sp_model_path=self.config.sp_model_path,
            dict_path=self.config.dict_path,
        )
        self.token_list = self.text_transform.token_list
        self.model: Optional[E2E] = None
        self.beam_search: Optional[BatchBeamSearch] = None
        self.preprocessor = VideoPreprocessor()
        self.model_loading_time: float = 0.0

    def load_model(self) -> E2E:
        """Instantiates the E2E Conformer model architecture."""
        t0 = time.time()
        self.model = E2E(
            odim=len(self.token_list),
            modality=self.config.modality,
            ctc_weight=self.config.ctc_weight,
        )
        self.model.to(self.device)
        self.model.eval()
        self.model_loading_time = time.time() - t0
        return self.model

    def load_checkpoint(self, checkpoint_path: Optional[Union[str, Path]] = None) -> None:
        """Loads pretrained weights into the Auto-AVSR model."""
        if checkpoint_path is None:
            checkpoint_path = self.config.checkpoint_path

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        t0 = time.time()
        if self.model is None:
            self.load_model()

        ckpt = torch.load(str(checkpoint_path), map_location=lambda storage, loc: storage)
        # Handle state_dict if nested
        state_dict = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.beam_search = get_beam_search_decoder(
            model=self.model,
            token_list=self.token_list,
            beam_size=self.config.beam_size,
            ctc_weight=self.config.ctc_weight,
            penalty=self.config.penalty,
            lm_weight=self.config.lm_weight,
        )
        self.model_loading_time += time.time() - t0

    def transcribe_tensor(
        self,
        tensor: torch.Tensor,
        use_beam_search: bool = True,
        return_details: bool = False,
        nbest: int = 10,
    ) -> Union[str, Dict[str, Any]]:
        """Transcribes preprocessed video tensor of shape (T, 1, 88, 88).

        Args:
            tensor: Preprocessed mouth ROI video tensor (T, 1, 88, 88)
            use_beam_search: Whether to run BatchBeamSearch or fast CTC greedy decode
            return_details: If True, returns rich dictionary with N-best candidates, scores,
                            confidence classification, and uncertainty margin.
            nbest: Maximum number of hypotheses to return

        Returns:
            str if return_details is False, else detailed Dict[str, Any].
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() or load_checkpoint() first.")

        tensor = tensor.to(self.device)
        with torch.no_grad():
            # Input to frontend: (B, T, C, H, W)
            x = self.model.frontend(tensor.unsqueeze(0))
            x = self.model.proj_encoder(x)
            enc_feat, _ = self.model.encoder(x, None)
            enc_feat = enc_feat.squeeze(0)  # (T, D)

            hypotheses: List[Dict[str, Any]] = []

            if use_beam_search and self.beam_search is not None:
                beam_hyps = self.beam_search(enc_feat)
                if beam_hyps:
                    for h in beam_hyps[:nbest]:
                        d = h.asdict()
                        token_seq = list(map(int, d["yseq"][1:]))
                        cand_text = self.text_transform.post_process(token_seq)
                        score_val = float(d["score"])
                        sub_scores = {k: float(v) for k, v in d.get("scores", {}).items()}
                        hypotheses.append({
                            "text": cand_text,
                            "tokens": token_seq,
                            "score": score_val,
                            "scores": sub_scores,
                        })

            # Fast CTC decoding fallback or supplementary hypotheses
            if not hypotheses:
                logits = self.model.ctc.ctc_lo(enc_feat)  # (T, odim)
                probs = torch.softmax(logits, dim=-1)
                best_tokens = torch.argmax(logits, dim=-1)
                
                # CTC collapse for top-1
                collapsed_1 = []
                prev = None
                for tok in best_tokens.cpu().tolist():
                    if tok != 0 and tok != prev:
                        collapsed_1.append(tok)
                    prev = tok
                text_1 = self.text_transform.post_process(collapsed_1)
                
                # Approximate sequence log-probability
                seq_log_prob = float(torch.sum(torch.log(torch.max(probs, dim=-1)[0] + 1e-12)).item())
                hypotheses.append({
                    "text": text_1,
                    "tokens": collapsed_1,
                    "score": round(seq_log_prob, 3),
                    "scores": {"ctc": round(seq_log_prob, 3)},
                })

                # Extract second-best path at high entropy frames for viseme alternative
                top2_tokens = torch.topk(logits, k=2, dim=-1).indices
                collapsed_2 = []
                prev = None
                for t_idx in range(best_tokens.size(0)):
                    tok = top2_tokens[t_idx, 1].item() if t_idx % 2 == 1 else top2_tokens[t_idx, 0].item()
                    if tok != 0 and tok != prev:
                        collapsed_2.append(tok)
                    prev = tok
                text_2 = self.text_transform.post_process(collapsed_2)
                if text_2 and text_2 != text_1:
                    hypotheses.append({
                        "text": text_2,
                        "tokens": collapsed_2,
                        "score": round(seq_log_prob - 2.5, 3),
                        "scores": {"ctc": round(seq_log_prob - 2.5, 3)},
                    })

            # Deduplicate hypotheses preserving highest score per unique text
            unique_hyps: List[Dict[str, Any]] = []
            seen_texts = set()
            for h in hypotheses:
                clean = h["text"].strip().upper()
                if clean and clean not in seen_texts:
                    seen_texts.add(clean)
                    unique_hyps.append(h)

            if not unique_hyps:
                unique_hyps = [{"text": "", "tokens": [], "score": -999.0, "scores": {}}]

            # Compute relative normalized softmax probabilities over candidates
            scores_tensor = torch.tensor([h["score"] for h in unique_hyps], dtype=torch.float32)
            temp = 2.0  # Temperature scaling for beam scores
            norm_probs = torch.softmax((scores_tensor - scores_tensor.max()) / temp, dim=0).tolist()
            for h, p in zip(unique_hyps, norm_probs):
                h["normalized_prob"] = round(p, 4)

            # Determine confidence and ambiguity margins
            top_score = unique_hyps[0]["score"]
            second_score = unique_hyps[1]["score"] if len(unique_hyps) > 1 else top_score - 10.0
            score_margin = round(top_score - second_score, 3)

            top_prob = unique_hyps[0]["normalized_prob"]
            top_text = unique_hyps[0]["text"]

            if top_score < -80.0 or len(top_text.split()) == 0:
                confidence_level = "UNCERTAIN"
                needs_clarification = True
            elif score_margin < 1.8 and len(unique_hyps) > 1:
                confidence_level = "AMBIGUOUS"
                needs_clarification = True
            else:
                confidence_level = "HIGH"
                needs_clarification = False

            # Extract distinct candidate phrases for user clarification
            clarification_candidates = [h["text"] for h in unique_hyps[:4]]

            if not return_details:
                return top_text

            return {
                "top_candidate": top_text,
                "hypotheses": unique_hyps,
                "confidence_level": confidence_level,
                "score_margin": score_margin,
                "top_probability": top_prob,
                "needs_clarification": needs_clarification,
                "clarification_candidates": clarification_candidates,
            }

    def transcribe(
        self,
        video_or_frames: Union[str, Path, np.ndarray, List[np.ndarray], torch.Tensor],
        use_beam_search: bool = True,
        nbest: int = 10,
    ) -> Dict[str, Any]:
        """Complete transcription pipeline for raw video, frame list, or preprocessed tensor.

        Returns comprehensive telemetry and hypothesis dictionary:
        - transcription: decoded top-1 text (maintains backward compatibility)
        - hypotheses: list of alternative hypotheses with scores and probabilities
        - confidence_level: 'HIGH', 'AMBIGUOUS', or 'UNCERTAIN'
        - score_margin: margin between top-1 and top-2 candidate
        - needs_clarification: whether uncertainty warrants user clarification
        - clarification_candidates: distinct candidate interpretations
        - inference_time: model forward time in seconds
        - preprocessing_time: MediaPipe + ROI + Normalization time in seconds
        - input_duration: video length in seconds
        - model_loading_time: initial loading time in seconds
        - total_latency: total execution time in seconds
        """
        start_total = time.time()
        t_prep_start = time.time()

        if isinstance(video_or_frames, (str, Path)):
            import cv2
            cap = cv2.VideoCapture(str(video_or_frames))
            frames = []
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            video_or_frames = frames
            input_duration = len(frames) / fps
        elif isinstance(video_or_frames, (np.ndarray, list)):
            input_duration = len(video_or_frames) / 25.0
        else:
            input_duration = video_or_frames.shape[0] / 25.0

        if isinstance(video_or_frames, torch.Tensor):
            tensor = video_or_frames
            t_prep = 0.0
        else:
            tensor, _ = self.preprocessor.process(video_or_frames)
            t_prep = time.time() - t_prep_start

        t_infer_start = time.time()
        details = self.transcribe_tensor(
            tensor,
            use_beam_search=use_beam_search,
            return_details=True,
            nbest=nbest,
        )
        t_infer = time.time() - t_infer_start
        total_latency = time.time() - start_total

        return {
            "transcription": details["top_candidate"],
            "hypotheses": details["hypotheses"],
            "confidence_level": details["confidence_level"],
            "score_margin": details["score_margin"],
            "top_probability": details["top_probability"],
            "needs_clarification": details["needs_clarification"],
            "clarification_candidates": details["clarification_candidates"],
            "inference_time": round(t_infer, 4),
            "preprocessing_time": round(t_prep, 4),
            "input_duration": round(input_duration, 4),
            "model_loading_time": round(self.model_loading_time, 4),
            "total_latency": round(total_latency, 4),
        }

