"""Minimal web UI for VoiceNo! V0.

Provides offline utterance transcription, calibration flow,
generic vs personalized comparison, and latency telemetry.
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import gradio as gr

from voiceno.calibration.prompts import CALIBRATION_PROMPTS, EVALUATION_PROMPTS
from voiceno.calibration.recorder import CalibrationRecorder
from voiceno.calibration.trainer import PersonalizationTrainer
from voiceno.clarification.manager import ClarificationManager
from voiceno.config import REPO_ROOT, VoiceNoConfig
from voiceno.context.rescorer import ContextRescorer, TaskContext
from voiceno.evaluation.evaluate import Evaluator
from voiceno.models.auto_avsr import AutoAVSRModel
from voiceno.models.personalization import PersonalizationManager



def build_ui(
    model_wrapper: Optional[AutoAVSRModel] = None,
    personalization_mgr: Optional[PersonalizationManager] = None,
    config: Optional[VoiceNoConfig] = None,
) -> gr.Blocks:
    """Constructs the Gradio interface for VoiceNo!."""
    config = config or VoiceNoConfig()

    if model_wrapper is None:
        model_wrapper = AutoAVSRModel(config.model)

    if personalization_mgr is None:
        personalization_mgr = PersonalizationManager(config.lora)

    recorder = CalibrationRecorder()
    context_rescorer = ContextRescorer()
    clarification_mgr = ClarificationManager()
    trainer = PersonalizationTrainer(model_wrapper, personalization_mgr, config.calibration, config.lora)
    evaluator = Evaluator(model_wrapper, personalization_mgr, context_rescorer)


    custom_css = """
    .brand-title { text-align: center; font-size: 2.2rem; font-weight: 800; margin-bottom: 2px; }
    .brand-tagline { text-align: center; font-size: 1.1rem; color: #6b7280; margin-bottom: 16px; font-style: italic; }
    .privacy-notice { text-align: center; font-size: 0.85rem; color: #10b981; margin-top: 10px; }
    """

    with gr.Blocks(title="VoiceNO! V0") as demo:
        gr.HTML(
            """
            <div class="brand-title">VoiceNO!</div>
            <div class="brand-tagline">You have the right to remain silent.</div>
            """
        )

        with gr.Row():
            status_box = gr.Textbox(
                label="System Status",
                value="Ready (Generic Mode)",
                interactive=False,
                scale=2,
            )
            mode_selector = gr.Radio(
                label="Recognition Mode",
                choices=["Generic", "Personalized"],
                value="Generic",
                interactive=True,
                scale=1,
            )
            active_profile = gr.Textbox(
                label="User Profile",
                value="default",
                interactive=True,
                scale=1,
            )

        with gr.Tabs():
            # TAB 1: Transcription
            with gr.TabItem("Transcribe Utterance"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(
                            label="Webcam / Video Utterance (Camera-Only)",
                            sources=["webcam", "upload"],
                        )
                        with gr.Accordion("🌐 Task Context (Optional Disambiguation Clues)", open=False):
                            ctx_app = gr.Dropdown(
                                label="Active Application",
                                choices=["None", "browser", "code_editor", "document_viewer", "calendar", "general"],
                                value="None",
                            )
                            ctx_keywords = gr.Textbox(
                                label="Context Keywords / Selected Text",
                                placeholder="e.g. mall, shopping, water, code, error",
                                lines=2,
                            )
                            apply_context_chk = gr.Checkbox(
                                label="Apply Contextual & Linguistic Rescoring",
                                value=True,
                            )
                        transcribe_btn = gr.Button("Transcribe Silent Utterance", variant="primary")
                    with gr.Column(scale=1):
                        output_text = gr.Textbox(
                            label="Top Candidate Interpretation",
                            placeholder="Transcribed text will appear here...",
                            lines=2,
                        )
                        confidence_badge = gr.Textbox(
                            label="Confidence & Ambiguity Status",
                            value="Awaiting utterance...",
                            lines=1,
                            interactive=False,
                        )
                        with gr.Accordion("🔍 N-Best Candidate Hypotheses & Probabilities", open=True):
                            nbest_box = gr.JSON(
                                label="Ranked Visual Speech Hypotheses",
                                value=[],
                            )
                        
                        # Interactive Clarification Panel
                        with gr.Group(visible=False) as clarif_panel:
                            gr.Markdown("### ❓ Visual Ambiguity Detected — Please Clarify:")
                            clarif_radio = gr.Radio(
                                label="Did you mean:",
                                choices=[],
                                interactive=True,
                            )
                            clarif_custom = gr.Textbox(
                                label="Or type what you silently articulated:",
                                placeholder="Type exact phrase...",
                                lines=1,
                            )
                            confirm_clarif_btn = gr.Button("Confirm Interpretation", variant="secondary")
                            clarif_status = gr.Textbox(
                                label="Clarification Event Status",
                                value="Awaiting choice...",
                                lines=1,
                                interactive=False,
                            )

                        telemetry_box = gr.JSON(
                            label="Performance & Latency Telemetry",
                            value={},
                        )


            # TAB 2: Calibration Flow
            with gr.TabItem("Calibration (Personalization)"):
                gr.Markdown(
                    """
                    ### 🎯 Personalization Calibration Flow
                    Silently articulate the prompted sentences displayed below.
                    VoiceNO will adapt its LoRA attention weights specifically for your articulatory patterns.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_display = gr.Textbox(
                            label="Calibration Prompt Sentence",
                            value=CALIBRATION_PROMPTS[0],
                            interactive=False,
                            lines=2,
                        )
                        calib_video = gr.Video(
                            label="Record Calibration Utterance (Silent)",
                            sources=["webcam", "upload"],
                        )
                        submit_calib_btn = gr.Button("Save Calibration Utterance", variant="secondary")
                        
                        strategy_choice = gr.Radio(
                            choices=[
                                ("⚡ Fast CPU Adapter (Instant, <50MB RAM, Recommended)", "fast_cpu"),
                                ("🏋️ Full Conformer LoRA (Heavy CPU / Cloud GPU)", "conformer_lora"),
                            ],
                            value="fast_cpu",
                            label="Calibration Strategy",
                            info="Fast CPU Adapter uses feature caching and optimizes in ~2s on low-spec CPUs without swap thrashing.",
                        )
                        epochs_slider = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1,
                            label="Calibration Epochs",
                        )
                        train_calib_btn = gr.Button("🚀 Train Personalization", variant="primary")
                        export_colab_btn = gr.Button("📦 Export Calibration Package (.zip) for Cloud / Colab", variant="secondary")
                        export_file_out = gr.File(label="Download Calibration Package (.zip)", interactive=False)
                    with gr.Column(scale=1):
                        calib_status = gr.Textbox(
                            label="Calibration Session Status",
                            value="No calibration data recorded yet for this session.",
                            lines=5,
                        )
                        calib_results = gr.JSON(
                            label="Training Telemetry",
                            value={},
                        )


            # TAB 3: Generic vs Personalized Evaluation
            with gr.TabItem("Evaluation (Held-out Benchmark)"):
                gr.Markdown(
                    """
                    ### 📊 Baseline vs Personalized Evaluation
                    Evaluates on held-out sentences never seen during calibration to measure actual WER & CER deltas.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        eval_prompt_display = gr.Textbox(
                            label="Held-Out Evaluation Sentence",
                            value=EVALUATION_PROMPTS[0],
                            interactive=False,
                            lines=2,
                        )
                        eval_video = gr.Video(
                            label="Record Evaluation Utterance",
                            sources=["webcam", "upload"],
                        )
                        save_eval_btn = gr.Button("Save Evaluation Sample", variant="secondary")
                        run_comparison_btn = gr.Button("Run Generic vs Personalized Evaluation", variant="primary")
                    with gr.Column(scale=1):
                        eval_report = gr.JSON(
                            label="Evaluation Report (WER / CER / Latency)",
                            value={},
                        )

        gr.HTML(
            """
            <div class="privacy-notice">
                🔒 <b>Privacy-First</b>: Strictly camera-only processing. Microphone is never accessed or recorded. Inference is entirely local.
            </div>
            """
        )

        # Session state to hold recorded utterances in memory
        session_calib_items = gr.State([])
        session_eval_items = gr.State([])

        # Event Handlers
        def on_mode_change(selected_mode, user_id):
            if selected_mode == "Personalized":
                try:
                    if model_wrapper.model is None:
                        model_wrapper.load_checkpoint()
                    profile_dir = REPO_ROOT / "user_profiles" / user_id
                    if (profile_dir / "lora.pt").is_file():
                        personalization_mgr.load_profile(model_wrapper.model, user_id=user_id)
                        return f"Personalized Mode active (Profile: {user_id})"
                    else:
                        personalization_mgr.remove_lora(model_wrapper.model)
                        return f"No profile found for '{user_id}'. Switched to Generic Mode."
                except Exception as e:
                    return f"Error loading profile: {e}"
            else:
                if model_wrapper.model is not None and personalization_mgr.is_personalized:
                    personalization_mgr.remove_lora(model_wrapper.model)
                return "Generic Mode active (Auto-AVSR Baseline)"

        mode_selector.change(
            fn=on_mode_change,
            inputs=[mode_selector, active_profile],
            outputs=[status_box],
        )

        session_transcription_state = gr.State({})

        def handle_transcription(video_path, mode, user_id, active_app, keywords_str, apply_context):
            if not video_path:
                return (
                    "Please provide a video recording or upload.",
                    "Awaiting utterance...",
                    [],
                    gr.update(visible=False),
                    gr.update(choices=[]),
                    "No video provided.",
                    {},
                    {},
                )

            try:
                # Ensure checkpoint loaded
                if model_wrapper.model is None:
                    model_wrapper.load_checkpoint()

                # Ensure correct personalization state
                if mode == "Personalized":
                    try:
                        personalization_mgr.load_profile(model_wrapper.model, user_id=user_id)
                    except Exception:
                        pass
                else:
                    if personalization_mgr.is_personalized:
                        personalization_mgr.remove_lora(model_wrapper.model)

                res = model_wrapper.transcribe(video_path)
                hyps = res.get("hypotheses", [{"text": res["transcription"], "score": 0.0}])

                # Apply Contextual Rescoring if requested
                if apply_context:
                    kws = [k.strip() for k in keywords_str.split(",") if k.strip()] if keywords_str else []
                    app_val = None if active_app == "None" else active_app
                    ctx = TaskContext(active_app=app_val, domain_keywords=kws)
                    hyps = context_rescorer.rescore_hypotheses(hyps, context=ctx)

                top_text = hyps[0]["text"] if hyps else res["transcription"]
                score_margin = res.get("score_margin", 0.0)
                conf = res.get("confidence_level", "HIGH")
                top_prob = hyps[0].get("rescored_prob", hyps[0].get("normalized_prob", 1.0))

                if conf == "HIGH":
                    badge = f"🟢 HIGH CONFIDENCE (Margin: {score_margin} | Prob: {top_prob*100:.1f}%)"
                    needs_clarif = False
                elif conf == "AMBIGUOUS":
                    badge = f"🟡 AMBIGUOUS — Close Alternatives Detected (Margin: {score_margin} | Prob: {top_prob*100:.1f}%)"
                    needs_clarif = True
                else:
                    badge = f"🔴 LOW CONFIDENCE / UNCERTAIN (Insufficient visual evidence)"
                    needs_clarif = True

                clarif_choices = [h["text"] for h in hyps[:4]] if hyps else [top_text]

                stored_state = {
                    "video_path": str(video_path),
                    "utterance_id": f"utt_{int(time.time() * 1000)}",
                    "top_text": top_text,
                    "hypotheses": hyps,
                    "confidence_level": conf,
                    "score_margin": score_margin,
                }

                return (
                    top_text,
                    badge,
                    hyps,
                    gr.update(visible=needs_clarif),
                    gr.update(choices=clarif_choices, value=clarif_choices[0] if clarif_choices else None),
                    "Ambiguity detected. Select intended candidate or write in below:" if needs_clarif else "Clear recognition. No clarification needed.",
                    res,
                    stored_state,
                )
            except Exception as e:
                err_msg = f"Transcription error: {str(e)}"
                return err_msg, "Error", [], gr.update(visible=False), gr.update(choices=[]), err_msg, {"error": str(e)}, {}

        transcribe_btn.click(
            fn=handle_transcription,
            inputs=[video_input, mode_selector, active_profile, ctx_app, ctx_keywords, apply_context_chk],
            outputs=[output_text, confidence_badge, nbest_box, clarif_panel, clarif_radio, clarif_status, telemetry_box, session_transcription_state],
        )

        def handle_clarification_confirm(user_id, selected_choice, custom_text, stored_state):
            chosen = custom_text.strip().upper() if custom_text and custom_text.strip() else (selected_choice or "")
            if not chosen:
                return "Please choose a candidate or type what you articulated."
            if not stored_state or not stored_state.get("hypotheses"):
                return "No active utterance to clarify."

            log_path = clarification_mgr.record_clarification(
                user_id=user_id,
                utterance_id=stored_state.get("utterance_id", f"utt_{int(time.time()*1000)}"),
                video_path=stored_state.get("video_path", ""),
                model_candidates=stored_state.get("hypotheses", []),
                chosen_interpretation=chosen,
                was_custom=bool(custom_text and custom_text.strip()),
                score_margin=stored_state.get("score_margin", 0.0),
                confidence_level=stored_state.get("confidence_level", "AMBIGUOUS"),
            )
            return f"Clarification confirmed: '{chosen}'. Logged to {log_path.name} (no silent retraining)."

        confirm_clarif_btn.click(
            fn=handle_clarification_confirm,
            inputs=[active_profile, clarif_radio, clarif_custom, session_transcription_state],
            outputs=[clarif_status],
        )


        def save_calib_utterance(video_path, current_prompt, user_id, current_items):
            if not video_path:
                return "Please record or upload a video.", current_prompt, current_items

            import cv2
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

            meta = recorder.save_utterance(
                user_id=user_id,
                frames=frames,
                prompt_text=current_prompt,
                partition="calibration",
            )
            updated_items = list(current_items) + [meta]

            # Rotate to next prompt
            next_idx = len(updated_items) % len(CALIBRATION_PROMPTS)
            next_prompt = CALIBRATION_PROMPTS[next_idx]
            status_msg = f"Saved {len(updated_items)} calibration utterance(s). Next prompt ready."

            return status_msg, next_prompt, updated_items

        submit_calib_btn.click(
            fn=save_calib_utterance,
            inputs=[calib_video, prompt_display, active_profile, session_calib_items],
            outputs=[calib_status, prompt_display, session_calib_items],
        )

        def run_lora_training(user_id, calib_items, strategy, epochs):
            if not calib_items:
                # Check if saved files exist on disk
                calib_items = recorder.load_dataset(user_id=user_id, partition="calibration")

            if not calib_items:
                return "No calibration data found to train on. Record at least one utterance first.", {}

            try:
                res = trainer.train_on_utterances(
                    user_id=user_id,
                    utterance_items=calib_items,
                    epochs=int(epochs),
                    strategy=strategy,
                )
                strategy_label = "Fast CPU Adapter" if strategy == "fast_cpu" else "Full Conformer LoRA"
                msg = (
                    f"Personalization training COMPLETE! Profile saved for '{user_id}'.\n"
                    f"• Strategy: {strategy_label}\n"
                    f"• Trainable parameters: {res['trainable_parameters']:,}\n"
                    f"• Optimization time: {res['training_seconds']}s\n"
                    f"• Status: Ready to transcribe in Personalized Mode!"
                )
                return msg, res
            except Exception as e:
                return f"Training error: {e}", {"error": str(e)}

        train_calib_btn.click(
            fn=run_lora_training,
            inputs=[active_profile, session_calib_items, strategy_choice, epochs_slider],
            outputs=[calib_status, calib_results],
        )

        def handle_colab_export(user_id):
            try:
                zip_path = recorder.export_calibration_package(user_id=user_id)
                return str(zip_path), f"Calibration package exported successfully to: {zip_path}"
            except Exception as e:
                return None, f"Export error: {e}"

        export_colab_btn.click(
            fn=handle_colab_export,
            inputs=[active_profile],
            outputs=[export_file_out, calib_status],
        )


        def save_eval_utterance(video_path, current_prompt, user_id, current_eval_items):
            if not video_path:
                return current_prompt, current_eval_items

            import cv2
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

            meta = recorder.save_utterance(
                user_id=user_id,
                frames=frames,
                prompt_text=current_prompt,
                partition="evaluation",
            )
            updated = list(current_eval_items) + [meta]
            next_idx = len(updated) % len(EVALUATION_PROMPTS)
            return EVALUATION_PROMPTS[next_idx], updated

        save_eval_btn.click(
            fn=save_eval_utterance,
            inputs=[eval_video, eval_prompt_display, active_profile, session_eval_items],
            outputs=[eval_prompt_display, session_eval_items],
        )

        def run_eval_comparison(user_id, eval_items):
            if not eval_items:
                eval_items = recorder.load_dataset(user_id=user_id, partition="evaluation")

            if not eval_items:
                return {"error": "No held-out evaluation samples available. Save at least one evaluation sample first."}

            try:
                report = evaluator.compare_generic_vs_personalized(
                    user_id=user_id,
                    evaluation_items=eval_items,
                )
                return report
            except Exception as e:
                return {"error": str(e)}

        run_comparison_btn.click(
            fn=run_eval_comparison,
            inputs=[active_profile, session_eval_items],
            outputs=[eval_report],
        )

    return demo
