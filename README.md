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

## Known limitations
- Case reports are published clinical narratives, not raw EHR or billing records; no costs/charges are annotated.
- Set-based scoring collapses repeated mentions within a chunk; spans crossing chunk boundaries are dropped (counted in `metrics.json`).
- Discontinuous spans are scored on their joined text; nested spans cannot be emitted by BIO taggers (count reported).

## Run
Kaggle -> New Notebook -> File -> Import `medrec_extraction_lora.ipynb` -> Settings: GPU T4 x2, Internet on -> Run All.
Outputs land in `results/` (`metrics.json`, `summary.csv`, `per_type.csv`, raw generations, LoRA adapter).

Repo: https://github.com/ByteBoss123/Medical-Record-Entity-Extraction-LoRA

Local tests (CPU, format fixture only): `pytest tests`.

## Results
Pending the Kaggle run.
