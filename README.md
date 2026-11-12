# Medical-Record Entity Extraction: LoRA Fine-Tuned LLM vs Prompting vs BioBERT

Extracts the facts a personal-injury medical chronology needs (diagnoses/injuries, symptoms,
body regions, diagnostic and therapeutic procedures, medications, dates, durations) from real
clinical narratives, and measures whether LoRA fine-tuning a small open LLM beats prompting it
and beats a fine-tuned biomedical encoder.

## Data
**MACCROBAT** (Caufield et al., 2019, figshare 10.6084/m9.figshare.9764942, CC BY 4.0):
200 PubMed Central clinical case reports with manual expert annotations (brat standoff).
The notebook downloads it directly; the 2020 release is preferred when both are present.

## Systems
| System | What it is |
|---|---|
| lexicon | longest-match lookup of train-set annotated strings |
| crf | sklearn-crfsuite BIO tagger, hand-built token features |
| biobert | `dmis-lab/biobert-base-cased-v1.2` token classifier, best epoch picked on val |
| qwen1.5b_zeroshot / _2shot | `Qwen2.5-1.5B-Instruct`, JSON extraction prompt |
| qwen1.5b_lora | same model + LoRA (r=16, all attention/MLP projections), 3 epochs, loss on answer tokens only |

## Evaluation (fixed before test results)
- Document-level 70/10/20 split, seed 13; 3-sentence chunks.
- Strict F1: exact (type, normalized text) match per chunk. Relaxed F1: same type and containment or token Jaccard >= 0.5, one-to-one.
- LLM-specific: JSON validity, truncation rate, hallucination rate (predicted strings not found in the source text).
- Paired document-level bootstrap (2,000 resamples) 95% CIs on micro-F1 differences.

## OCR robustness (simulated scans)
Each test chunk is rendered as a page image at three pre-set degradation levels (clean / mild / heavy:
blur, speckle noise, skew, JPEG, downscale; `ocr.py`), OCR'd with Tesseract, and re-scored against the
original gold labels. Reports character error rate and the F1 drop per system with bootstrap CIs.
Degradation is synthetic, applied to real text; these are not real scanned records.

## Serving on Kubernetes
`serve/` holds a FastAPI service (`POST /extract` -> typed entities with character offsets) backed by the CRF
model. The `k8s-deploy` GitHub Actions workflow trains the CRF on real MACCROBAT, builds the Docker image,
deploys 2 replicas to a kind cluster with readiness/liveness probes, smoke-tests offsets through the Service,
and runs a 500-request load test (results uploaded as the `k8s-run` artifact). The LLM is not served here:
the serving demo uses the CPU model so it runs on a free CI runner.

## Known limitations
- Case reports are published clinical narratives, not raw EHR or billing records; no costs/charges are annotated.
- Set-based scoring collapses repeated mentions within a chunk; spans crossing chunk boundaries are dropped (counted in `metrics.json`).
- Discontinuous spans are scored on their joined text; nested spans cannot be emitted by BIO taggers (count reported).

## Run
Kaggle -> New Notebook -> File -> Import `medrec_extraction_lora.ipynb` -> Settings: GPU T4 x2, Internet on -> Run All.
Outputs land in `results/` (`metrics.json`, `summary.csv`, `per_type.csv`, raw generations, LoRA adapter).

Repo: https://github.com/ByteBoss123/Medical-Record-Entity-Extraction-LoRA

Local tests (CPU, format fixture only): `pytest tests`.
Local service: `python serve/train_crf.py && uvicorn serve.app:app`.

## Results
### Verified in CI (GitHub Actions run 43301762015, Nov 12 2026)
- Data: MACCROBAT2020, 200 documents; split 140 / 20 / 40 (train / val / test documents).
- CRF baseline on the 40 test documents: strict micro-F1 0.575, macro-F1 0.537 (8 entity types).
- Kubernetes (kind, 2 replicas, readiness/liveness probes): rollout succeeded; 500/500 `/extract` requests
  returned 200 through the Service; latency p50 11.8 ms, p95 46.7 ms, p99 48.6 ms
  (CI runner via `kubectl port-forward`, ~2,000-character inputs; not a production benchmark).

### LLM / BioBERT / OCR comparison
Pending the Kaggle run.
