"""Assemble the Kaggle notebook from cell sources. Run: python build_notebook.py"""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).parent
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Medical-record entity extraction: LoRA fine-tuned LLM vs prompting vs BioBERT
Real data: **MACCROBAT** (Caufield et al., 2019; CC BY 4.0) — 200 PubMed Central clinical case reports
with manual expert annotations. Task: pull the facts a personal-injury medical chronology needs
(diagnoses/injuries, symptoms, body regions, diagnostic and therapeutic procedures, medications, dates, durations).

**Pre-registered setup (fixed before seeing test results):** document-level 70/10/20 split (seed 13);
3-sentence chunks; strict + relaxed set-based F1 per chunk; paired document-level bootstrap (2,000 resamples).
Run on **Kaggle, GPU T4 x2 (one GPU used), Internet ON**. Runtime ≈ 2–2.5 h.
""")

code("""
!pip -q uninstall -y torchao  # Kaggle ships torchao 0.10; current peft rejects <0.16 and we do not use it
!pip -q install -U peft sklearn-crfsuite
import torch, transformers, peft, sys, platform
print(torch.__version__, transformers.__version__, peft.__version__, platform.python_version())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU — enable T4 in Settings")
""")

code("%%writefile mre.py\n" + (HERE / "mre.py").read_text())

code(r'''
# ---- Download MACCROBAT from figshare (article 9764942) and load ----
import json, os, zipfile, io, urllib.request, glob, pathlib, collections
import mre, importlib; importlib.reload(mre)
os.makedirs("data", exist_ok=True); os.makedirs("results", exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (research notebook)"}
get = lambda u: urllib.request.urlopen(urllib.request.Request(u, headers=UA)).read()
api = json.loads(get("https://api.figshare.com/v2/articles/9764942"))
print("figshare version:", api.get("version"), [f["name"] for f in api["files"]])
def extract(blob, dest):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for n in z.namelist():
            if n.endswith(".zip"):
                extract(z.read(n), os.path.join(dest, pathlib.Path(n).stem))
            elif not n.endswith("/"):
                p = os.path.join(dest, n); os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, "wb").write(z.read(n))
for f in api["files"]:
    blob = get(f["download_url"])
    if f["name"].endswith(".zip"):
        extract(blob, os.path.join("data", pathlib.Path(f["name"]).stem))
# Prefer the 2020 release (improved annotation consistency) if both are present
ann_dirs = sorted({os.path.dirname(p) for p in glob.glob("data/**/*.ann", recursive=True)})
print("annotation dirs:", ann_dirs)
pick = [d for d in ann_dirs if "2020" in d] or ann_dirs
docs = []
for d in pick: docs += mre.load_brat_dir(d)
seen=set(); docs=[x for x in docs if not (x.doc_id in seen or seen.add(x.doc_id))]
cnt = collections.Counter(s.type for d in docs for s in d.spans)
stats = {"source_dirs": pick, "n_docs": len(docs), "n_spans": sum(cnt.values()),
         "spans_per_type": dict(cnt),
         "discontinuous_spans": sum(s.discontinuous for d in docs for s in d.spans)}
print(json.dumps(stats, indent=1))
''')

code(r'''
# ---- Split and chunk ----
sp = mre.split_docs(docs, seed=13)
CH = {k: [c for d in v for c in mre.chunk_doc(d)] for k, v in sp.items()}
cross = {k: sum(mre.cross_boundary_spans(d, mre.chunk_doc(d)) for d in v) for k, v in sp.items()}
stats.update({"split_docs": {k: len(v) for k, v in sp.items()},
              "split_chunks": {k: len(v) for k, v in CH.items()},
              "spans_lost_at_chunk_boundaries": cross,
              "test_doc_ids": sorted(d.doc_id for d in sp["test"])})
print({k: stats[k] for k in ("split_docs", "split_chunks", "spans_lost_at_chunk_boundaries")})
test = CH["test"]; RESULTS = {}; PRED = {}
def record(name, preds, extra=None):
    PRED[name] = preds
    RESULTS[name] = {"strict": mre.score(test, preds), "relaxed": mre.score(test, preds, relaxed=True), **(extra or {})}
    r = RESULTS[name]
    print(f"{name:28s} strict micro-F1 {r['strict']['micro_f1']:.3f} | relaxed {r['relaxed']['micro_f1']:.3f}")
''')

code(r'''
# ---- Baseline 1: train-set lexicon lookup ----
lex = mre.build_lexicon(CH["train"])
record("lexicon", {(c.doc_id, c.idx): mre.lexicon_predict(c, lex) for c in test}, {"lexicon_size": len(lex)})
''')

code(r'''
# ---- Baseline 2: CRF (sklearn-crfsuite) BIO tagger ----
import sklearn_crfsuite
def xy(chs):
    X, Y, drop = [], [], 0
    for c in chs:
        toks, tags, d = mre.bio_labels(c); drop += d
        X.append(mre.crf_features(toks)); Y.append(tags)
    return X, Y, drop
Xtr, Ytr, drop_tr = xy(CH["train"])
crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=200, all_possible_transitions=True)
crf.fit(Xtr, Ytr)
pc = {}
for c in test:
    toks = mre.tokenize(c.text)
    pc[(c.doc_id, c.idx)] = mre.tags_to_set(c.text, toks, crf.predict_single(mre.crf_features(toks)))
record("crf", pc, {"train_spans_not_bio_encodable": drop_tr})
''')

code(r'''
# ---- Baseline 3: BioBERT token classifier (fine-tuned, best epoch on val) ----
import numpy as np, random, time, torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
torch.manual_seed(0); random.seed(0); np.random.seed(0)
BERT = "dmis-lab/biobert-base-cased-v1.2"
LABELS = ["O"] + [f"{p}-{t}" for t in mre.TYPES for p in "BI"]; L2I = {l: i for i, l in enumerate(LABELS)}
btok = AutoTokenizer.from_pretrained(BERT)
def enc(c):
    e = btok(c.text, truncation=True, max_length=512, return_offsets_mapping=True)
    offs = e["offset_mapping"]; lab = [-100 if s == t == 0 else 0 for s, t in offs]
    for sp_ in sorted(c.spans, key=lambda s: -(s.end - s.start)):
        idx = [i for i, (s, t) in enumerate(offs) if not s == t == 0 and s >= sp_.start and t <= sp_.end]
        if not idx or any(lab[i] != 0 for i in idx): continue
        lab[idx[0]] = L2I["B-" + sp_.type]
        for i in idx[1:]: lab[i] = L2I["I-" + sp_.type]
    return e["input_ids"], lab, offs
def bert_predict(model, chs, bs=32):
    model.eval(); out = {}
    for i in range(0, len(chs), bs):
        b = chs[i:i+bs]; e = btok([c.text for c in b], truncation=True, max_length=512, padding=True,
                                  return_offsets_mapping=True, return_tensors="pt")
        offs = e.pop("offset_mapping")
        with torch.no_grad(): lg = model(**{k: v.cuda() for k, v in e.items()}).logits.argmax(-1).cpu()
        for c, o, p, m in zip(b, offs, lg, e["attention_mask"]):
            toks, tags = [], []
            for (s, t), y, mm in zip(o.tolist(), p.tolist(), m.tolist()):
                if mm and not s == t == 0: toks.append((c.text[s:t], s, t)); tags.append(LABELS[y])
            out[(c.doc_id, c.idx)] = mre.tags_to_set(c.text, toks, tags)
    return out
tr = [enc(c) for c in CH["train"]]
bert = AutoModelForTokenClassification.from_pretrained(BERT, num_labels=len(LABELS)).cuda()
opt = torch.optim.AdamW(bert.parameters(), lr=3e-5, weight_decay=0.01)
best, best_state, curve = -1, None, []
t0 = time.time()
for ep in range(10):
    bert.train(); random.shuffle(tr)
    for i in range(0, len(tr), 8):
        b = tr[i:i+8]; n = max(len(x[0]) for x in b)
        ids = torch.tensor([x[0] + [btok.pad_token_id] * (n - len(x[0])) for x in b]).cuda()
        lab = torch.tensor([x[1] + [-100] * (n - len(x[1])) for x in b]).cuda()
        loss = bert(input_ids=ids, attention_mask=(ids != btok.pad_token_id).long(), labels=lab).loss
        loss.backward(); opt.step(); opt.zero_grad()
    vf = mre.score(CH["val"], bert_predict(bert, CH["val"]))["micro_f1"]; curve.append(vf)
    print(f"epoch {ep+1} val strict micro-F1 {vf:.3f}")
    if vf > best: best, best_ep, best_state = vf, ep + 1, {k: v.detach().cpu().clone() for k, v in bert.state_dict().items()}
bert.load_state_dict(best_state)
t1 = time.time(); pb = bert_predict(bert, test); lat = (time.time() - t1) / len(test)
record("biobert", pb, {"best_epoch": best_ep, "val_curve": curve, "train_minutes": (t1 - t0) / 60,
                       "sec_per_chunk": lat})
# keep `bert` loaded: reused in the OCR robustness cell
''')

code(r'''
# ---- LLM: Qwen2.5-1.5B-Instruct, zero-shot and 2-shot prompting ----
from transformers import AutoModelForCausalLM
LLM = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(LLM); tok.padding_side = "left"
llm = AutoModelForCausalLM.from_pretrained(LLM, torch_dtype=torch.float16).cuda().eval()
# 2 fixed few-shot examples: the two shortest train chunks that contain >= 4 spans
shots = sorted([c for c in CH["train"] if len(c.spans) >= 4], key=lambda c: len(c.text))[:2]
def messages(c, fewshot=False):
    m = [{"role": "system", "content": mre.SYSTEM}]
    if fewshot:
        for s in shots: m += [{"role": "user", "content": mre.instruction(s.text)},
                              {"role": "assistant", "content": mre.target_json(s)}]
    return m + [{"role": "user", "content": mre.instruction(c.text)}]
def generate(model, chs, fewshot=False, bs=16, max_new=512):
    raw_out, sets, raws, valid, trunc = {}, {}, {}, 0, 0
    t0 = time.time()
    for i in range(0, len(chs), bs):
        b = chs[i:i+bs]
        prompts = [tok.apply_chat_template(messages(c, fewshot), tokenize=False, add_generation_prompt=True) for c in b]
        e = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = model.generate(**e, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        for c, seq in zip(b, g[:, e["input_ids"].shape[1]:]):
            txt = tok.decode(seq, skip_special_tokens=True); k = (c.doc_id, c.idx)
            trunc += int((seq != tok.eos_token_id).sum().item() >= max_new)
            s, r, ok = mre.parse_llm_json(txt); valid += ok
            raw_out[k], sets[k], raws[k] = txt, s, r
    n = len(chs)
    return sets, raws, raw_out, {"json_valid_rate": valid / n, "truncated_rate": trunc / n,
                                 "sec_per_chunk": (time.time() - t0) / n}
for name, fs in [("qwen1.5b_zeroshot", False), ("qwen1.5b_2shot", True)]:
    s, r, raw, ex = generate(llm, test, fewshot=fs)
    ex["hallucination_rate"] = mre.hallucination_rate(test, r)
    record(name, s, ex)
    json.dump({f"{k[0]}#{k[1]}": v for k, v in raw.items()}, open(f"results/raw_{name}.json", "w"))
''')

code(r'''
# ---- LoRA fine-tune (PEFT) on train chunks; loss on assistant tokens only ----
from peft import LoraConfig, get_peft_model
torch.manual_seed(0); random.seed(0)
lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(llm, lcfg)
for n_, p in model.named_parameters():
    if p.requires_grad: p.data = p.data.float()  # fp32 adapters, fp16 frozen base
model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.config.use_cache = False
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
def sft_example(c):
    prompt = tok.apply_chat_template(messages(c), tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tok(mre.target_json(c) + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p_ids + t_ids)[:1024]
    return ids, ([-100] * len(p_ids) + t_ids)[:1024]
train_ex = [sft_example(c) for c in CH["train"]]
EPOCHS, ACC, LR = 3, 8, 2e-4
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
total = EPOCHS * len(train_ex) // ACC
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / total))
scaler = torch.amp.GradScaler("cuda"); losses = []; t0 = time.time(); step = 0
model.train()
for ep in range(EPOCHS):
    random.shuffle(train_ex); run = 0.0
    for i, (ids, lab) in enumerate(train_ex):
        x = torch.tensor([ids]).cuda(); y = torch.tensor([lab]).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=x, labels=y).loss / ACC
        scaler.scale(loss).backward(); run += loss.item()
        if (i + 1) % ACC == 0:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step(); step += 1
    losses.append(run * ACC / len(train_ex)); print(f"epoch {ep+1} mean train loss {losses[-1]:.4f}")
train_min = (time.time() - t0) / 60
model.gradient_checkpointing_disable(); model.eval(); model.config.use_cache = True
vs, vr, _, vex = generate(model, CH["val"])
val_f1 = mre.score(CH["val"], vs)["micro_f1"]
s, r, raw, ex = generate(model, test)
ex.update({"hallucination_rate": mre.hallucination_rate(test, r), "train_minutes": train_min,
           "epoch_train_loss": losses, "trainable_params": trainable, "val_strict_micro_f1": val_f1,
           "hparams": {"r": 16, "alpha": 32, "dropout": 0.05, "epochs": EPOCHS, "lr": LR, "grad_accum": ACC, "max_len": 1024}})
record("qwen1.5b_lora", s, ex)
json.dump({f"{k[0]}#{k[1]}": v for k, v in raw.items()}, open("results/raw_qwen1.5b_lora.json", "w"))
model.save_pretrained("results/lora_adapter")
''')

md("""
## OCR robustness (simulated scans)
Real medical records often arrive as scans. Each test chunk is rendered as a page image at three pre-set
degradation levels (blur, speckle noise, skew, JPEG, downscale), OCR'd with Tesseract, and re-scored against
the **original** gold annotations. Degradation is synthetic, applied to real text; these are not real scans.
""")

code("!apt-get -qq install -y tesseract-ocr > /dev/null 2>&1; pip -q install pytesseract; tesseract --version | head -1")
code("%%writefile ocr.py\n" + (HERE / "ocr.py").read_text())

code(r'''
import ocr, importlib; importlib.reload(ocr)
from multiprocessing.pool import ThreadPool  # threads, not fork: forking a CUDA/torch process can deadlock; tesseract runs as a subprocess anyway
os.environ["OMP_THREAD_LIMIT"] = "1"  # one OpenMP thread per tesseract call
os.makedirs("results/ocr_samples", exist_ok=True)
jobs = [((c.doc_id, c.idx), c.text, lv, i) for i, c in enumerate(test) for lv in ocr.LEVELS]
t0 = time.time()
with ThreadPool(4) as pool: outs = pool.map(ocr.ocr_job, jobs, chunksize=8)
print(f"OCR {len(jobs)} pages in {(time.time()-t0)/60:.1f} min")
for lv in ocr.LEVELS: ocr.render(test[0].text, lv, 0).save(f"results/ocr_samples/{test[0].doc_id}_{lv}.jpg")
OCR = {lv: {} for lv in ocr.LEVELS}; CER = {lv: [] for lv in ocr.LEVELS}
for k, lv, txt, c in outs: OCR[lv][k] = txt; CER[lv].append(c)
OCR_RESULTS = {}; OCR_PRED = {}
for lv in ocr.LEVELS:
    och = [mre.Chunk(c.doc_id, c.idx, c.start, c.end, OCR[lv][(c.doc_id, c.idx)], c.spans) for c in test]
    preds = {"lexicon": {(c.doc_id, c.idx): mre.lexicon_predict(c, lex) for c in och}}
    pc = {}
    for c in och:
        toks = mre.tokenize(c.text); pc[(c.doc_id, c.idx)] = mre.tags_to_set(c.text, toks, crf.predict_single(mre.crf_features(toks)))
    preds["crf"] = pc
    preds["biobert"] = bert_predict(bert, och)
    s_, r_, _, ex_ = generate(model, och); preds["qwen1.5b_lora"] = s_
    OCR_PRED[lv] = preds
    OCR_RESULTS[lv] = {"mean_cer": float(np.mean(CER[lv])), "median_cer": float(np.median(CER[lv])),
                       "lora_json_valid_rate": ex_["json_valid_rate"],
                       "systems": {n_: {m: mre.score(test, p_, relaxed=(m == "relaxed"))["micro_f1"]
                                        for m in ("strict", "relaxed")} for n_, p_ in preds.items()}}
    print(lv, f"CER {OCR_RESULTS[lv]['mean_cer']:.3f}",
          {n_: round(v["strict"], 3) for n_, v in OCR_RESULTS[lv]["systems"].items()})
OCR_BOOT = {f"{n_}_{lv}_minus_original": mre.paired_bootstrap(test, OCR_PRED[lv][n_], PRED[n_])
            for lv in ("mild", "heavy") for n_ in ("biobert", "qwen1.5b_lora")}
print(json.dumps({k: [round(v["diff"], 4), [round(x, 4) for x in v["ci95"]]] for k, v in OCR_BOOT.items()}, indent=1))
''')

code(r'''
# ---- Paired bootstrap, tables, save everything ----
import pandas as pd
pairs = [("qwen1.5b_lora", "qwen1.5b_2shot"), ("qwen1.5b_lora", "qwen1.5b_zeroshot"),
         ("qwen1.5b_lora", "biobert"), ("biobert", "crf"), ("crf", "lexicon")]
boot = {f"{a}_minus_{b}": {m: mre.paired_bootstrap(test, PRED[a], PRED[b], relaxed=(m == "relaxed"))
                           for m in ("strict", "relaxed")} for a, b in pairs}
rows = []
for n_, r_ in RESULTS.items():
    rows.append({"system": n_, **{f"{m}_{k}": round(r_[m][k], 4) for m in ("strict", "relaxed")
                                  for k in ("micro_precision", "micro_recall", "micro_f1", "macro_f1")},
                 **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r_.items()
                    if k in ("json_valid_rate", "truncated_rate", "hallucination_rate", "sec_per_chunk")}})
summary = pd.DataFrame(rows); summary.to_csv("results/summary.csv", index=False)
per = pd.DataFrame([{"system": n_, "type": t, **{f"strict_{k}": round(v, 4) if isinstance(v, float) else v
                     for k, v in r_["strict"]["per_type"][t].items()},
                     "relaxed_f1": round(r_["relaxed"]["per_type"][t]["f1"], 4)}
                    for n_, r_ in RESULTS.items() for t in mre.TYPES])
per.to_csv("results/per_type.csv", index=False)
env = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
       "gpu": torch.cuda.get_device_name(0)}
json.dump({"data": stats, "results": RESULTS, "bootstrap": boot, "ocr": OCR_RESULTS,
           "ocr_bootstrap": OCR_BOOT, "env": env},
          open("results/metrics.json", "w"), indent=1, default=float)
display(summary)
print(json.dumps({k: {m: {"diff": round(v[m]["diff"], 4), "ci95": [round(x, 4) for x in v[m]["ci95"]]}
                      for m in v} for k, v in boot.items()}, indent=1))
!cd results && zip -qr ../results.zip . && ls -la ..
''')

md("""
**Send back:** `results/metrics.json`, `results/summary.csv`, `results/per_type.csv`
(or the whole `results.zip` from the Output panel). Nothing is claimed until these numbers come back.
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "accelerator": "GPU"}
nbf.write(nb, HERE / "medrec_extraction_lora.ipynb")
print("cells:", len(cells))
