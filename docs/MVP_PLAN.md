# Memact: zero-budget MVP plan

Updated: 21 October 2026. Status: proposed engineering and validation plan; performance targets below are not measured results.

Implementation checkpoint: the first evaluation repairs are implemented locally. Model-state checks, strict adapter loading, recorded-prediction analysis, failure accounting, corpus-weighted oracle WER, and a portable Colab baseline workflow now have 36 passing CPU tests. The fabricated greedy alternative has been removed. No actual remote GPU accuracy or training run has been performed. Start with [the baseline quickstart](BASELINE_QUICKSTART.md); capture/FPS validation, product UX, learned rescoring, and larger datasets remain outstanding. The audit below describes the initial state and motivates the work.

## 1. Product decision

Build silent prompts into AI tools first. The long-term direction is an input method for computers, agents, and text fields. The first usable experience is: record a short, silently mouthed instruction, review the transcript, correct or choose an alternative when needed, and copy it into an existing AI tool.

Initial audience: English-speaking AI power users working from shared spaces, with a front-facing laptop webcam and internet access. Start with seated, reasonably frontal use and short prompts of roughly 3–8 seconds. Test this scope; do not describe it as universal dictation. Support new sentence combinations rather than requiring memorized demo commands. Proper nouns, exact identifiers, numbers, and negation need explicit evaluation.

Constraints agreed with founder: $0 recurring spend, no local heavy model inference, Colab available for experiments. No required paid API, paid GPU, credit card, or new hardware. Preserve the current prototype while replacing untrustworthy pieces incrementally.

The first demo can live inside a notebook. A public demo can follow if genuinely free hosting is available. A dependable desktop input method is a later release with an explicit compute dependency. A conference-quality pitch needs a live workflow, honest measured evidence, and a credible path to repeated use; it does not require implementing every future input surface.

## 2. What the repository actually establishes

| Area | Evidence in code | Decision |
| --- | --- | --- |
| Visual recognition | Auto-AVSR wrapper, bundled ESPnet architecture, checkpoint present locally | Keep as the first baseline behind a replaceable backend |
| Preprocessing | MediaPipe landmarks, affine alignment, mouth crops, normalization | Keep, then verify against the checkpoint's expected pipeline and actual capture timing |
| Personalization | CTC head adapter and attention LoRA; profile save/load | Keep as experiments, measure on unseen sentences and a later session |
| Viseme workarounds | Manually specified word swaps, article replacements, n-gram scores, domain regex bonuses | Retain only as a labeled heuristic baseline |
| Neural language scorer | Optional causal LM likelihood with lazy loading | Benchmark remotely; do not require an additional large model initially |
| Evaluation | Useful metric utilities and comparative evaluator, but unreliable experiment orchestration | Repair before making accuracy claims |
| Desktop integration | Fixed-duration camera CLI; missing `List` import; no implemented system hotkey or cursor insertion | Build after recognition and compute gates |
| Agent execution | UI constructs action JSON, then returns a success string without dispatch | Replace with truthful review/copy behavior |
| Cloud privacy | Mouth payload API; ordinary HTTP; deletion of Python tensor references | Remove enclave, anonymity, and guaranteed memory-wipe claims |

Specific evidence problems to fix:

- `experiments/run_open_vocab_eval.py` labels Condition B personalized without loading the adapter before that run. The `mode` argument in `Evaluator.evaluate_utterances` labels a report; it does not switch weights.
- Its `perform_qualitative_failure_analysis` returns hand-authored example results without performing inference. These must be labeled illustrative or replaced with analysis of actual saved predictions.
- The end-to-end test assigns unrelated calibration/evaluation prompts to slices of a benchmark video. It can exercise plumbing, but cannot establish transcription accuracy or personalization benefit.
- Existing context tests use phrases also encoded into scoring tables. They demonstrate those rules, not generalization.
- The greedy fallback fabricates a second path by alternating top-two frame tokens and assigns an arbitrary score offset. Treat only the best path as a greedy result; implement actual CTC prefix search if alternatives are needed.
- Normalized beam probabilities are conditional on the candidate set and are not calibrated correctness probabilities. One bad candidate can receive probability 1.
- `target_fps` is configured but the current transcription path does not resample input video to it. Measure timestamps and correct frame rate before comparing models.
- Uncompressed payload serialization loses quantization metadata; file-path input to the edge extractor is declared but passed into a frame-only preprocessing method.

Prior inspection ran 21 unit tests successfully. No GPU quality or full training result has been established in this session.

The founder notes cite LISA's 28.8% → 12.2% result as camera lip-reading evidence. The cited paper actually reports that result on silent EMG, using muscle sensors. Its language-assisted decoding is an inspiration, not a Memact performance prediction. Vocalized-video benchmarks also do not establish accuracy on silent mouthing. See sources below.

## 3. Architecture within $0

### Guaranteed starting workflow, subject to Colab GPU availability

Use an interactive Colab notebook for setup, explicit capture/upload, GPU inference, optional training, evaluation, and transcript review. Keep compute calls inside notebook execution. Do not build a hidden desktop API, public tunnel, keepalive mechanism, or always-on service on free Colab.

The laptop runs capture, lightweight quality checks, and eventually mouth cropping. Colab runs the visual model and decoding. The user reviews text and copies it into their chosen AI tool, using its existing account. No agent API key is required.

For the first research baseline, uploading video to the user's own notebook is acceptable only with an accurate disclosure that the video leaves the device. The target privacy architecture uploads mouth crops instead. First support locally extracted crop packages; then implement browser-side extraction after validating Python/browser crop parity. Do not claim local-only full-face processing while full recordings are still uploaded.

Crop transport should use bounded uint8 arrays with a versioned schema: frame shape/count, capture timestamps, effective frame rate, preprocessing version, request identifier, and optional profile identifier. Normalize once at the backend. At 25 FPS, 88×88 uint8 crops are about 194 kB/second before compression; a 2-second clip is about 387 kB. Payload size depends on duration and content.

Separate capture, preprocessing, recognition, candidate ranking, decision policy, and delivery. Give notebook and hosted adapters the same input/output contract. The client should not import PyTorch or load Auto-AVSR merely to start recording.

Return actual candidates with visual scores, ranking scores, model/profile/preprocessing versions, timing breakdowns, and one of: `review`, `choose_alternative`, `rerecord`, `unavailable`. Treat content as data. Copying reviewed text is the first real integration; sending or executing it remains an explicit user action.

### Conditional free public demo

Investigate Hugging Face ZeroGPU as the first hosted option. Current official documentation says free personal accounts in good standing, with verified email and older than 30 days, can host up to two ZeroGPU Gradio Spaces. It lists daily GPU quotas and queues. Account eligibility, deployment compatibility, and checkpoint rights still need verification; no account has been checked or deployment created here.

If eligible, package the same inference backend as a bounded Gradio GPU function. Keep queue, quota exhaustion, cold start, and unavailable states visible. Check current official runtime requirements when implementing; the current model wrapper's CUDA detection/loading behavior needs testing in ZeroGPU's execution model. Start with one shared generic model; add isolated profiles only after proving safe separation.

If eligibility or compatibility fails, continue the notebook demo. Apply for a community GPU grant after producing evidence. Grants are optional opportunities, not assumed infrastructure. Never silently fall back to billable hardware.

### Privacy language we can earn

Camera-only, no microphone request, explicit capture indicator, stop/cancel, encrypted transport where hosted, no application-level recording retention by default, and consent-based research saving are realistic engineering commitments. Mouth crops remain sensitive visual data; cropping is data minimization, not anonymity. Notebook outputs, temporary uploads, provider logs, profile storage, and saved benchmark packages each need an explicit retention policy. TEE deployment and attestation are later work, not properties supplied by Colab or a generic GPU server.

## 4. Rebuild the viseme strategy around evidence

Visual ambiguity is a limitation of the observed signal. A better language prior can help choose among visually plausible sequences, but can also confidently substitute the wrong intent. The product must expose uncertainty and make correction cheap.

Run the following in order on identical held-out clips:

1. Generic visual model with real beam output, no hand-written substitutions.
2. That model with a trained language scorer. Tune score weight and length effects on development data; cache model outputs so reranking does not rerun the encoder.
3. The best configuration with genuine user-provided task context. Compare correct, absent, and misleading context. Context must exist before recording; never generate keywords from the reference transcript.
4. Personalization at 0, 1, 3, and 5 minutes of calibration. Compare the head adapter and attention LoRA separately. Test on new sentences and another day. The head adapter changes output mapping; it does not adapt visual features.
5. Only if candidate coverage is poor, evaluate a second model or a stronger visual adaptation method. Investigate newer VSR work as candidates, not automatic upgrades.

Use a score such as `visual_score + lambda * language_score + beta * context_score + length_term`, with all tunable terms selected on development data. Avoid counting the same language prior twice without an ablation. An N-best list is not a full token lattice. Word-swap expansions are generated proposals, not visually scored paths; if retained, rescore them against the video with a compatible scoring method and label their origin.

Measure both top-1 errors and oracle top-k errors. If the intended text is absent from candidates, reranking alone cannot recover it. If it is present but loses, investigate ranking. If it remains ambiguous, ask the user. Do not use free-form LLM rewriting to manufacture a polished prompt and report it as faithful transcription.

Preserve negation, numbers, entities, and requested operations in evaluation. Test contrasts such as “run the tests” / “do not run the tests” and “explain this” / “delete this.” Include blank capture, no face, ordinary nonspeech mouth motion, and unsupported prompts.

## 5. Measurement and decision gates

First collect a small founder dataset: 20 calibration utterances and 60 held-out utterances over at least two sessions. Include short natural AI prompts, unfamiliar combinations, names/numbers, ambiguity cases, and negative captures. Keep an independently set-aside test subset; tune on development examples only. Prompt text is an intended label, so review failed/partial articulations rather than assuming every recording matches its prompt.

Then recruit 5–10 consenting testers for two sessions each, roughly 30 evaluation clips per session plus a separate calibration set. Begin with 3–5 testers if recruitment is difficult. Record lighting, device, pose, frame quality, and whether speech was genuinely silent. Use new speakers to evaluate generic performance and new sessions to evaluate personalized stability. Track prompt overlap explicitly. A several-hundred-clip pilot is directional evidence, not population-level validation.

Every run stores: code revision, checkpoint hash, profile hash, split manifest, preprocessing settings, decoder settings, model versions, actual predictions, and error/status records. Never omit failed clips from the denominator. Store consented evaluation artifacts separately from ordinary product use.

Report raw WER/CER, sentence accuracy, oracle top-k WER, intent preservation judged against a predefined rubric, review/correction rate, false confident outputs, no-speech false positives, failures, and release-to-review p50/p95 latency. Report results per speaker/session and uncertainty intervals where meaningful. Separate queue, upload, inference, ranking, and delivery time. For usefulness, measure total time including recording and correction against typing the same type of prompt, with task order varied.

Provisional release targets, to be revisited after the baseline rather than represented as achieved:

| Gate | Proposed evidence |
| --- | --- |
| Research baseline | One reproducible Colab run; real generic/personalized switching; measured predictions and timings; no illustrative outcomes mixed into results |
| Founder demo | At least 80% of held-out short prompts usable without retyping the whole sentence; record exact transcription separately; show failures and correction burden |
| Small pilot | At least 90% intent-preserving completion within one correction or rerecord; report all attempts and per-user differences |
| Responsiveness | Aim for warm median release-to-review ≤3 seconds and p95 ≤6 seconds; report hosted queue/cold-start results separately; free infrastructure may miss these |
| Confident suggestions | Tune an abstention policy on validation data and report error versus coverage; a future goal of ≤2% wrong accepted suggestions requires substantially more evidence than this pilot |
| Usefulness | Users voluntarily reuse it on later days; measure corrected completion time and preference, even if it does not yet beat typing |

If recognition misses the founder gate, stop adding distribution features and investigate crop/timing fidelity, silent-speech mismatch, and candidate coverage. If recognition works but users dislike mouthing publicly, change the interaction or audience. If it works but free compute is too slow, retain a research demo and seek sponsored compute before promising a daily utility.

## 6. Delivery sequence

Timing is indicative active engineering time, conditional on GPU access, recordings, and measured progress. This is not a promise that model quality can be solved on a fixed calendar.

| Stage | Work | Exit artifact |
| --- | --- | --- |
| A: 2–4 days | Repair evaluation switching and illustrative reporting; fix capture/payload/CLI defects; separate lightweight client dependencies; document checkpoint provenance | Trustworthy benchmark command, minimal client checks, exact Colab setup |
| B: 3–7 days plus data collection | Capture founder data, validate crop/FPS pipeline, run baseline, save real N-best outputs and timing | Baseline report and diagnosis of visual versus ranking errors |
| C: 1–2 weeks | Test trained rescoring and personalization ablations; implement explicit quality/uncertainty decisions | Measured improvement or a documented failed hypothesis |
| D: about 1 week | Notebook capture/review/copy flow; bounded payloads; clear errors; optional eligible ZeroGPU deployment | Live silent prompt → reviewed text → AI tool workflow |
| E: 1–2 weeks plus recruitment | Test with other users in ordinary shared spaces; tune UX on development participants; freeze final evaluation | Pilot report, unedited demo, observed repeat use |

Later, after the gates: a browser extension or small desktop companion adds a global shortcut, focus-safe insertion, cancellation, accessibility integration, and per-OS packaging. Universal keyboard/IME support, mobile keyboards, multilingual dictation, streaming model training, and confidential GPU infrastructure each need separate projects. Start with one OS/application integration when compute permits it.

## 7. First implementation backlog

Execute these sequentially; each should leave a reviewable change:

1. Make evaluation switch actual model/profile state and assert the active checkpoint/profile in results. Replace invented case results with categories derived from predictions. Add meaningful regression tests for switching and report provenance.
2. Fix desktop import, beam-size handling, uncompressed payload round trips, path input, and frame timestamp/FPS handling. Validate payload dimensions, duration, finite values, and decompressed size before inference.
3. Add a deterministic recorded-crop benchmark path that runs on Colab, uses the same Python code as normal inference, and exports actual JSON predictions. Keep a small CPU-only fake backend for client/error-flow checks; never confuse it with real inference.
4. Clean the master notebook: revision-pinned setup, hardware/checkpoint verification, explicit recording consent, actual model state, measured timing, saved-output controls, and notebook-native inference. Remove reliance on public `--share` serving for the free path.
5. Collect founder calibration/development/test clips and run the baseline. This requires the founder's real silent recordings and an available remote GPU; it cannot be validated through synthetic unit tests.
6. Cache real hypotheses, then evaluate language scoring and adaptation. Promote only a configuration that improves held-out results without unacceptable intent substitution.
7. Build the review/copy experience and verify free hosting eligibility, compatibility, and model rights before public deployment.

Do not wholesale rewrite the bundled model before obtaining a clean baseline. Preserve a known working reference and replace one layer at a time.

## 8. Startup evidence and rights

The launch story: “Prompt your AI silently using the webcam you already own.” Demonstrate several previously unseen short requests, one ambiguity correction, and a real response from an existing AI tool. State the supported capture conditions, setup, latency, compute arrangement, and failure rate. Label any fallback recording as recorded. Do not promise a particular conference acceptance or fundraising outcome.

Interview 5–10 target users while testing: when they avoid speaking to AI, how often it matters, whether mouthing feels acceptable, how much correction is tolerable, and whether they return without prompting. Maintain an opt-in tester list and collect willingness to pay after the workflow works; do not infer demand from demo excitement alone.

The potential defensible asset is consented silent-mouthing data, cross-session personalization evidence, reliable correction behavior, and integration quality. A wrapper plus regex rules is not sufficient differentiation.

Before distributing or commercially deploying weights, trace the exact 3,291-hour checkpoint, bundled assets, tokenizer, and training-data terms. Upstream Auto-AVSR explicitly distinguishes its Apache-licensed code from possible model/data terms. The local repository license alone does not settle those rights. Record what is known and obtain clarification where needed; do not declare the checkpoint commercially cleared or categorically prohibited without that evidence.

## Sources checked on 21 October 2026

- [MONA LISA paper](https://arxiv.org/html/2403.05583v1): the cited 12.2% WER result concerns silent EMG, not webcam lip-reading.
- [Visual-only recognition of normal, whispered and silent speech](https://arxiv.org/abs/1802.06399): evidence of differences between vocalized and silent speech conditions; not a Memact accuracy estimate.
- [Auto-AVSR upstream](https://github.com/mpc001/auto_avsr): architecture/baseline reference and distinction between code and pretrained-model terms.
- [Colab FAQ](https://research.google.com/colaboratory/faq.html): free interactive notebook compute, variable resource availability, and restrictions on unrelated hosting and bypassing notebook use.
- [ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu): conditional free hosting eligibility, supported execution model, quotas, and queues.
- [Spaces overview](https://huggingface.co/docs/hub/spaces-overview): ZeroGPU exception for eligible free accounts and optional community GPU grants.
