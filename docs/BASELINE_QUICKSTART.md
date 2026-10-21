# Run the first honest Memact baseline

This workflow uses your laptop for preparing files and your own interactive Colab GPU notebook for inference. It starts no public server and requires no paid API. GPU availability on free Colab is variable.

## 1. Prepare the current source on your laptop

From the project root, activate a Python environment with `requirements.txt` installed. See the README for environment setup.

```bash
python experiments/package_eval_source.py
```

This creates `memact_eval_source.zip` with current source and preprocessing/tokenizer assets. It excludes user recordings, user profiles, the large VSR checkpoint, and founder notes. This works before changes are pushed to GitHub.

Export your existing recordings separately:

```bash
python -m memact export-calib --user default
```

That produces `calibration_data/default_calibration_package.zip`. Unlike the source bundle, this archive contains your recorded videos. Uploading it sends those videos to your Colab runtime. Review the recordings and labels first. Existing calibration/evaluation folders do not establish label correctness or independence across sessions.

For the first useful baseline, use about 10–20 genuinely silent, clearly articulated evaluation clips with accurate labels. A single existing clip is a pipeline check, not an accuracy claim. Later expand the independent dataset described in `MVP_PLAN.md`.

## 2. Open the notebook on Colab

Upload `experiments/Memact_Baseline.ipynb` through Colab's **File → Upload notebook**. Select a GPU runtime. Run the cells in order, upload the source archive, and then upload the calibration package when requested. The notebook stops if no GPU is available; it does not fall back to heavy laptop inference.

The notebook downloads the existing checkpoint and verifies the expected MD5 from the project setup. It does not establish the checkpoint's commercial-use rights. The report also records the loaded checkpoint's SHA-256, source manifest, dataset hashes, adapter state, settings, and actual predictions.

The initial baseline needs no profile and does not train. Treat current confidence scores as uncalibrated. Read per-recording failures and candidate lists alongside aggregate WER and exact match.

## 3. Compare personalization only after training a profile

Upload `lora.pt` and `metadata.json` through the optional notebook cell. It will run:

```bash
python experiments/run_open_vocab_eval.py --user default --personalized
```

The runner verifies that an adapter was actually loaded. Invalid/incomplete profiles are rejected. Generic and personalized predictions are saved, and context rescoring reuses the personalized candidates. The oracle score uses reference labels and is an upper bound on candidate recovery, not delivered recognition accuracy. Current context rescoring remains a labeled heuristic baseline.

## 4. Bring the report back

Download the resulting JSON with the last cell. It contains reference text and predictions, so avoid posting it publicly if the content is sensitive. Clear notebook outputs and delete uploaded/extracted recordings when finished. Provider and notebook storage behavior is separate from application cleanup; this workflow makes no enclave or secure-erasure claim.

No remote GPU evaluation has been run as part of the local harness repair. Local tests exercise correctness with synthetic tiny models and cannot measure speech recognition quality.
