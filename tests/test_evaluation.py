"""CPU-only checks of evaluation integrity, not recognition accuracy."""

import copy
import json

import pytest
import torch
from torch import nn

from experiments.run_open_vocab_eval import build_evaluation_items, run_open_vocab_evaluation
from memact.config import LoRAConfig, ModelConfig
from memact.evaluation.analysis import analyze_predictions
from memact.evaluation.evaluate import Evaluator
from memact.models.personalization import LoRALinear, PersonalizationManager
from memact.models.auto_avsr import AutoAVSRModel
from types import SimpleNamespace


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_q = nn.Linear(2, 2)


class StubRecognizer:
    """Deliberately synthetic output whose choice depends on real adapter weights."""

    def __init__(self):
        self.model = TinyModel()
        self.config = ModelConfig()
        self.checkpoint_info = {"path": "synthetic-test-backend", "sha256": "test-only"}
        self.calls = []

    def transcribe(self, path, **kwargs):
        layer = self.model.linear_q
        adapted = isinstance(layer, LoRALinear) and bool(torch.all(layer.lora_B == 0.5))
        self.calls.append(adapted)
        if str(path).endswith("broken.mp4"):
            raise RuntimeError("Unreadable recording")
        top = "CONVERT THIS IMAGE TO PDF" if adapted else "WRONG WORDS"
        return {
            "transcription": top,
            "hypotheses": [{"text": top, "score": -1.0}, {"text": "OTHER WORDS", "score": -3.0}],
            "inference_time": 0.02,
            "total_latency": 0.03,
            "needs_clarification": False,
            "score_margin": 2.0,
        }


@pytest.fixture
def setup_eval(tmp_path):
    wrapper = StubRecognizer()
    manager = PersonalizationManager(LoRAConfig(profiles_dir=tmp_path / "profiles"))
    manager.apply_lora(wrapper.model)
    with torch.no_grad():
        for layer in manager.lora_layers.values():
            layer.lora_B.fill_(0.5)
    manager.save_profile("speaker")
    manager.remove_lora(wrapper.model)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"synthetic fixture, not a video")
    items = [{"utterance_id": "clip", "video_path": str(video), "prompt": "CONVERT THIS IMAGE TO PDF"}]
    return wrapper, manager, items


def test_comparison_switches_real_weights_and_reuses_visual_candidates(setup_eval, tmp_path):
    wrapper, manager, items = setup_eval
    # Even with adapters initially active, Condition A must run the base model.
    manager.load_profile(wrapper.model, "speaker")
    evaluator = Evaluator(wrapper, manager)
    report = evaluator.evaluate_four_conditions("speaker", items, tmp_path / "reports")
    assert wrapper.calls == [False, True]  # D performs no extra inference.
    conditions = report["conditions"]
    assert conditions["A_generic_baseline"]["wer"] > 0
    assert conditions["B_personalized"]["wer"] == 0
    state = conditions["B_personalized"]["recognition_state"]
    assert state["user_id"] == "speaker"
    assert state["adapter_sha256"]
    assert state["profile"]["sha256"]
    assert conditions["D_personalized_context"]["visual_results_reused"]
    assert conditions["C_personalized_nbest_oracle"]["oracle_only"]
    assert "personalization_improved" in report["failure_analysis"][0]["categories"]
    assert report["provenance"]["dataset"][0]["video_sha256"]
    saved = json.loads((tmp_path / "reports" / "open_vocab_eval_speaker.json").read_text())
    assert saved == report


def test_mode_label_cannot_misrepresent_adapter_state(setup_eval):
    wrapper, manager, items = setup_eval
    evaluator = Evaluator(wrapper, manager)
    with pytest.raises(ValueError, match="active adapters"):
        evaluator.evaluate_utterances(items, mode="Personalized")
    manager.load_profile(wrapper.model, "speaker")
    with pytest.raises(ValueError, match="Generic evaluation"):
        evaluator.evaluate_utterances(items, mode="Generic")
    # A second manager must not claim ownership of the first manager's adapters.
    with pytest.raises(ValueError, match="owned by this manager"):
        Evaluator(wrapper).evaluate_utterances(items, mode="Personalized")
    assert not wrapper.calls


def test_failures_remain_in_denominator(setup_eval):
    wrapper, manager, items = setup_eval
    manager.load_profile(wrapper.model, "speaker")
    items.append({"utterance_id": "broken", "video_path": "broken.mp4", "prompt": "CONVERT THIS IMAGE TO PDF"})
    result = Evaluator(wrapper, manager).evaluate_utterances(items, mode="Personalized")
    assert result["num_utterances"] == 2
    assert result["num_failed"] == 1
    assert result["wer"] == 0.5
    assert result["exact_match"] == 0.5
    assert result["utterances"][1]["status"] == "error"
    assert "Unreadable recording" in result["utterances"][1]["error"]


def test_failed_no_speech_clip_is_not_counted_as_correct(setup_eval):
    wrapper, manager, _ = setup_eval
    result = Evaluator(wrapper, manager).evaluate_utterances([
        {"video_path": "broken.mp4", "prompt": ""},
    ])
    assert result["exact_match"] == 0
    assert result["num_failed"] == 1


def test_cached_outputs_reject_changed_labels_or_weights(setup_eval):
    wrapper, manager, items = setup_eval
    manager.load_profile(wrapper.model, "speaker")
    evaluator = Evaluator(wrapper, manager)
    visual = evaluator.evaluate_utterances(items, mode="Personalized")
    changed = copy.deepcopy(items)
    changed[0]["prompt"] = "A DIFFERENT REFERENCE"
    with pytest.raises(ValueError, match="evaluation items"):
        evaluator.evaluate_utterances(changed, mode="Personalized", apply_context=True, visual_results=visual)
    with torch.no_grad():
        wrapper.model.linear_q.lora_B.add_(0.1)
    with pytest.raises(ValueError, match="active model"):
        evaluator.evaluate_utterances(items, mode="Personalized", apply_context=True, visual_results=visual)


def test_reject_empty_and_duplicate_evaluation_sets(setup_eval):
    wrapper, manager, items = setup_eval
    evaluator = Evaluator(wrapper, manager)
    with pytest.raises(ValueError, match="empty run"):
        evaluator.evaluate_utterances([])
    with pytest.raises(ValueError, match="unique"):
        evaluator.evaluate_utterances(items + items)


def test_uninitialized_checkpoint_is_rejected(setup_eval):
    wrapper, manager, items = setup_eval
    wrapper.checkpoint_info = None
    with pytest.raises(RuntimeError, match="loaded checkpoint"):
        Evaluator(wrapper, manager).evaluate_utterances(items)


def test_analysis_includes_regressions_without_claiming_viseme_causes():
    def condition(text, wer):
        return {"utterances": [{"utterance_id": "x", "reference": "RUN TESTS", "hypothesis": text,
                                "wer": wer, "exact_match": float(wer == 0), "nbest": [text], "status": "ok"}]}
    cases = analyze_predictions(condition("RUN TESTS", 0), condition("RUN", 0.5), condition("DELETE FILES", 1))
    assert cases[0]["categories"] == ["generic_correct", "personalization_regressed", "rescoring_regressed", "unresolved_error"]
    assert cases[0]["evidence_source"] == "recorded_predictions"


def test_dataset_is_not_padded_with_bundled_samples(tmp_path):
    with pytest.raises(ValueError, match="empty run"):
        build_evaluation_items("speaker", data_dir=tmp_path)


def test_dataset_rejects_missing_files_and_calibration_overlap(tmp_path):
    root = tmp_path / "speaker"
    evaluation = root / "evaluation"
    calibration = root / "calibration"
    evaluation.mkdir(parents=True)
    calibration.mkdir()
    item = {"utterance_id": "test", "prompt": "RUN TESTS", "video_path": "missing.mp4"}
    (evaluation / "test.json").write_text(json.dumps(item))
    with pytest.raises(FileNotFoundError, match="Missing evaluation"):
        build_evaluation_items("speaker", tmp_path)
    (evaluation / "test.mp4").write_bytes(b"duplicate clip")
    (calibration / "train.mp4").write_bytes(b"duplicate clip")
    (calibration / "train.json").write_text(json.dumps(item))
    with pytest.raises(ValueError, match="both calibration and evaluation"):
        build_evaluation_items("speaker", tmp_path)


def test_runner_refuses_implicit_cpu_inference(tmp_path, monkeypatch):
    evaluation = tmp_path / "speaker" / "evaluation"
    evaluation.mkdir(parents=True)
    (evaluation / "test.mp4").write_bytes(b"fixture")
    (evaluation / "test.json").write_text(json.dumps({"video_path": "test.mp4", "prompt": "RUN TESTS"}))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Colab GPU"):
        run_open_vocab_evaluation("speaker", data_dir=tmp_path)


def test_greedy_fallback_does_not_fabricate_alternative_candidates():
    # Bypass model construction/checkpoint loading. Only exercise the decoder
    # using a tiny fixed tensor, never the actual visual recognition model.
    wrapper = object.__new__(AutoAVSRModel)
    wrapper.device = torch.device("cpu")
    wrapper.beam_search = lambda features: []
    wrapper.text_transform = SimpleNamespace(post_process=lambda tokens: " ".join(map(str, tokens)))
    logits = torch.tensor([[0., 5., 2.], [0., 5., 2.], [5., 0., 2.], [0., 1., 5.]])
    wrapper.model = SimpleNamespace(
        frontend=lambda tensor: tensor,
        proj_encoder=lambda tensor: tensor,
        encoder=lambda tensor, mask: (logits.unsqueeze(0), None),
        ctc=SimpleNamespace(ctc_lo=lambda features: features),
    )
    result = wrapper.transcribe_tensor(torch.zeros(4, 1, 2, 2), return_details=True)
    assert result["decoding_method"] == "ctc_greedy"
    assert len(result["hypotheses"]) == 1
    assert result["hypotheses"][0]["tokens"] == [1, 2]
    assert result["hypotheses"][0]["score_kind"] == "greedy_alignment_log_probability"
