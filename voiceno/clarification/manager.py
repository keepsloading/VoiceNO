"""Clarification mechanism and interaction logger for VoiceNO.

Handles ambiguous recognition outcomes, formats user-facing disambiguation choices,
and records explicit clarification events without automatic retraining.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

from voiceno.config import REPO_ROOT


@dataclass
class ClarificationRecord:
    """Record of an interactive user clarification event."""
    user_id: str
    utterance_id: str
    video_path: str
    model_candidates: List[Dict[str, Any]]
    chosen_interpretation: str
    was_custom_write_in: bool
    score_margin: float
    confidence_level: str
    context_used: Optional[Dict[str, Any]] = None
    timestamp: int = 0


class ClarificationManager:
    """Manages ambiguity resolution and logs explicit user corrections."""

    def __init__(self, log_dir: Optional[Union[str, Path]] = None):
        self.log_dir = Path(log_dir or (REPO_ROOT / "clarification_logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def extract_clarification_choices(
        self,
        hypotheses: List[Dict[str, Any]],
        max_choices: int = 3,
    ) -> List[str]:
        """Extracts concise candidate choices for user clarification."""
        choices = []
        seen = set()
        for h in hypotheses:
            txt = h.get("text", "").strip()
            if txt and txt not in seen:
                seen.add(txt)
                choices.append(txt)
            if len(choices) >= max_choices:
                break
        return choices

    def record_clarification(
        self,
        user_id: str,
        utterance_id: str,
        video_path: str,
        model_candidates: List[Dict[str, Any]],
        chosen_interpretation: str,
        was_custom: bool = False,
        score_margin: float = 0.0,
        confidence_level: str = "AMBIGUOUS",
        context: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Appends a user clarification event to the user-specific jsonl log.

        Does NOT retrain the model silently.
        """
        user_log_dir = self.log_dir / user_id
        user_log_dir.mkdir(parents=True, exist_ok=True)
        log_file = user_log_dir / "clarifications.jsonl"

        record = ClarificationRecord(
            user_id=user_id,
            utterance_id=utterance_id,
            video_path=str(video_path),
            model_candidates=model_candidates,
            chosen_interpretation=chosen_interpretation.strip().upper(),
            was_custom_write_in=was_custom,
            score_margin=score_margin,
            confidence_level=confidence_level,
            context_used=context,
            timestamp=int(time.time()),
        )

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")

        return log_file

    def get_user_clarifications(self, user_id: str) -> List[Dict[str, Any]]:
        """Reads historical clarification events for a user."""
        log_file = self.log_dir / user_id / "clarifications.jsonl"
        if not log_file.is_file():
            return []

        entries = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
