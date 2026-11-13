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

### Extraction quality (Kaggle T4, Nov 12 2026; 40 held-out test reports, 8 entity types)
| System | Strict precision | Strict recall | Strict micro-F1 | Strict macro-F1 |
|---|---|---|---|---|
| lexicon | 0.413 | 0.418 | 0.415 | 0.356 |
| crf | 0.619 | 0.537 | 0.575 | 0.537 |
| biobert (fine-tuned) | 0.618 | 0.699 | **0.656** | 0.619 |
| qwen1.5b_zeroshot | 0.321 | 0.178 | 0.229 | 0.232 |
| qwen1.5b_2shot | 0.301 | 0.263 | 0.281 | 0.299 |
| qwen1.5b_lora | 0.639 | 0.644 | **0.642** | 0.604 |

Relaxed micro-F1: zero-shot 0.379, 2-shot 0.430, BioBERT 0.787, LoRA 0.765.
LoRA training loss by epoch: 0.195, 0.100, 0.054.

Paired document-level bootstrap (2,000 resamples), strict micro-F1 difference [95% CI]:
- LoRA minus 2-shot prompting: **+0.361** [0.318, 0.397]
- LoRA minus BioBERT: -0.014 [-0.033, 0.004], **a statistical tie on strict F1**; on relaxed F1 BioBERT is ahead (-0.022 [-0.038, -0.005])
- BioBERT minus CRF: +0.081 [0.054, 0.106]
- CRF minus lexicon: +0.160 [0.137, 0.185]

### More baselines on the same split (`experiments/cpu_models.py`, Nov 13 2026)
| System | Strict micro-F1 | Run |
|---|---|---|
| spaCy NER (blank `en`, 30 epochs, best epoch on val) | 0.550 | GitHub Actions `spacy-only`, CPU |
| FLAN-T5-small seq2seq (15 epochs, linearized `type: text` targets) | 0.597 | Kaggle T4 (`seq2seq_gpu.ipynb`) |

Paired bootstrap vs the CRF (0.575): spaCy -0.025 [-0.040, -0.011]; FLAN-T5 +0.021 [+0.005, +0.036].
Both runs are logged with Weights & Biases (offline mode unless `WANDB_API_KEY` is set).

### OCR robustness (simulated scans, same 40 test reports)
| Level | Mean CER | lexicon | crf | biobert | qwen1.5b_lora |
|---|---|---|---|---|---|
| clean render | 0.3% | 0.406 | 0.567 | 0.648 | 0.639 |
| mild | 5.3% | 0.378 | 0.432 | 0.484 | **0.512** |
| heavy | 27.2% | 0.298 | 0.257 | 0.280 | **0.337** |

Drop from original text (strict micro-F1, paired bootstrap 95% CI):
BioBERT mild -0.172 [-0.189, -0.153], heavy -0.376 [-0.398, -0.351];
LoRA mild -0.129 [-0.148, -0.110], heavy -0.305 [-0.326, -0.282].
The fine-tuned LLM loses less accuracy than BioBERT as OCR noise rises. A direct LoRA-vs-BioBERT
significance test under OCR was not run, so the ranking under noise is descriptive.
Note: the heavy level produced 27% CER on real case-report text, well above the ~10% it gave on the
hand-written calibration sample; levels were not re-tuned after seeing results.

Run notes: the run hit two environment issues, both fixed in this repo: Kaggle's preinstalled
torchao 0.10 blocked the current peft (now uninstalled first), and a forked multiprocessing pool
hung in a CUDA process (now a thread pool).
