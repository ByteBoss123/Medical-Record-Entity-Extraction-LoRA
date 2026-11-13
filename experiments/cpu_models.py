"""CPU experiments on the same MACCROBAT split: spaCy NER and a FLAN-T5 seq2seq extractor.

Runs on a CPU (GitHub Actions). Tracks runs with Weights & Biases: online when
WANDB_API_KEY is set, otherwise offline (run files saved under wandb/).

Usage: python experiments/cpu_models.py [--data DIR] [--smoke]
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "serve"))
import mre  # noqa: E402

SEP, KSEP = " | ", ": "


# ------------------------------------------------------------- data
def load_split(data_dir):
    if data_dir:
        dirs = [data_dir]
    else:
        from train_crf import download
        dirs = download()
    docs, seen = [], set()
    for d in dirs:
        for doc in mre.load_brat_dir(d):
            if doc.doc_id not in seen:
                seen.add(doc.doc_id)
                docs.append(doc)
    sp = mre.split_docs(docs, seed=13)
    return {k: [c for d in v for c in mre.chunk_doc(d)] for k, v in sp.items()}, dirs, len(docs)


# ------------------------------------------------------------ spaCy
def spacy_docs(nlp, chunks):
    from spacy.tokens import DocBin
    from spacy.util import filter_spans
    db, dropped = DocBin(), 0
    for c in chunks:
        doc = nlp.make_doc(c.text)
        ents = []
        for s in c.spans:
            if s.discontinuous:
                dropped += 1
                continue
            span = doc.char_span(s.start, s.end, label=s.type, alignment_mode="contract")
            if span is None:
                dropped += 1
            else:
                ents.append(span)
        kept = filter_spans(ents)
        dropped += len(ents) - len(kept)
        doc.ents = kept
        db.add(doc)
    return db, dropped


def run_spacy(CH, epochs, log):
    import spacy
    from spacy.training import Example
    random.seed(0)
    spacy.util.fix_random_seed(0)
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    for t in mre.TYPES:
        ner.add_label(t)
    db, dropped = spacy_docs(nlp, CH["train"])
    train = [Example(nlp.make_doc(d.text), d) for d in db.get_docs(nlp.vocab)]
    opt = nlp.initialize(lambda: train)
    best, best_bytes, curve = -1, None, []
    t0 = time.time()
    for ep in range(epochs):
        random.shuffle(train)
        losses = {}
        for batch in spacy.util.minibatch(train, size=16):
            nlp.update(batch, sgd=opt, drop=0.2, losses=losses)
        vf = mre.score(CH["val"], spacy_predict(nlp, CH["val"]))["micro_f1"]
        curve.append(vf)
        log({"spacy/epoch": ep + 1, "spacy/train_loss": losses.get("ner", 0.0), "spacy/val_strict_micro_f1": vf})
        print(f"spaCy epoch {ep+1} loss {losses.get('ner', 0):.1f} val F1 {vf:.3f}", flush=True)
        if vf > best:
            best, best_ep, best_bytes = vf, ep + 1, nlp.to_bytes()
    nlp.from_bytes(best_bytes)
    preds = spacy_predict(nlp, CH["test"])
    return preds, {"best_epoch": best_ep, "val_curve": curve, "train_spans_not_encodable": dropped,
                   "train_minutes": (time.time() - t0) / 60, "spacy_version": spacy.__version__}


def spacy_predict(nlp, chunks):
    out = {}
    for c, doc in zip(chunks, nlp.pipe([c.text for c in chunks])):
        out[(c.doc_id, c.idx)] = {(e.label_, mre.norm(e.text)) for e in doc.ents if mre.norm(e.text)}
    return out


# ---------------------------------------------------------- seq2seq
def linearize(chunk):
    seen, parts = set(), []
    for s in sorted(chunk.spans, key=lambda s: s.start):
        k = (s.type, s.text)
        if k not in seen:
            seen.add(k)
            parts.append(f"{s.type}{KSEP}{s.text}")
    return SEP.join(parts) if parts else "none"


def parse_linear(s):
    out = set()
    for part in s.split(SEP.strip()):
        if KSEP.strip() not in part:
            continue
        t, txt = part.split(KSEP.strip(), 1)
        t = t.strip()
        if t in mre.TYPES and mre.norm(txt):
            out.add((t, mre.norm(txt)))
    return out


def run_seq2seq(CH, epochs, log, model_name="google/flan-t5-small", smoke=False):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    torch.manual_seed(0)
    random.seed(0)
    torch.set_num_threads(os.cpu_count() or 2)
    tok = AutoTokenizer.from_pretrained(model_name)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(dev)
    prefix = "Extract medical entities (type: exact text): "
    train = CH["train"][:40] if smoke else CH["train"]
    src = [prefix + c.text for c in train]
    tgt = [linearize(c) for c in train]
    tl = [len(tok(t)["input_ids"]) for t in tgt]
    max_tgt = min(384, max(tl))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    bs, t0, curve = 8, time.time(), []
    idx = list(range(len(src)))
    for ep in range(epochs):
        model.train()
        random.shuffle(idx)
        tot = 0.0
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            enc = tok([src[j] for j in b], max_length=512, truncation=True, padding=True, return_tensors="pt")
            lab = tok([tgt[j] for j in b], max_length=max_tgt, truncation=True, padding=True, return_tensors="pt")["input_ids"]
            lab[lab == tok.pad_token_id] = -100
            loss = model(**{k: v.to(dev) for k, v in enc.items()}, labels=lab.to(dev)).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            tot += loss.item() * len(b)
        tr = tot / len(idx)
        log({"seq2seq/epoch": ep + 1, "seq2seq/train_loss": tr})
        print(f"seq2seq epoch {ep+1} loss {tr:.4f} ({(time.time()-t0)/60:.1f} min)", flush=True)
        curve.append(tr)
    train_min = (time.time() - t0) / 60
    model.eval()
    vchunks = CH["val"][:20] if smoke else CH["val"]
    tchunks = CH["test"][:20] if smoke else CH["test"]
    vp, _ = s2s_predict(model, tok, vchunks, prefix, max_tgt, dev)
    t1 = time.time()
    tp, raw = s2s_predict(model, tok, tchunks, prefix, max_tgt, dev)
    return tp, raw, {"model": model_name, "epochs": epochs, "lr": 5e-4, "batch_size": bs,
                     "epoch_train_loss": curve, "train_minutes": train_min,
                     "val_strict_micro_f1": mre.score(vchunks, vp)["micro_f1"],
                     "sec_per_chunk": (time.time() - t1) / len(tchunks), "max_target_tokens": max_tgt, "device": dev}, tchunks


def s2s_predict(model, tok, chunks, prefix, max_tgt, dev="cpu", bs=16):
    import torch
    out, raw = {}, {}
    for i in range(0, len(chunks), bs):
        b = chunks[i:i + bs]
        enc = tok([prefix + c.text for c in b], max_length=512, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            g = model.generate(**{k: v.to(dev) for k, v in enc.items()}, max_new_tokens=max_tgt, num_beams=1)
        for c, s in zip(b, tok.batch_decode(g, skip_special_tokens=True)):
            out[(c.doc_id, c.idx)] = parse_linear(s)
            raw[(c.doc_id, c.idx)] = s
    return out, raw


# -------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-seq2seq", action="store_true")
    ap.add_argument("--skip-spacy", action="store_true")
    ap.add_argument("--out", default="results_cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    CH, dirs, n_docs = load_split(a.data)

    import wandb
    mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
    run = wandb.init(project="medical-record-entity-extraction", name="cpu-spacy-seq2seq", mode=mode,
                     config={"split_seed": 13, "n_docs": n_docs, "smoke": a.smoke, "types": mre.TYPES})
    log = run.log

    test = CH["test"]
    crf_preds = {}
    import pickle
    mp = ROOT / "serve" / "model" / "crf.pkl"
    if mp.exists():
        crf = pickle.load(open(mp, "rb"))
        for c in test:
            toks = mre.tokenize(c.text)
            crf_preds[(c.doc_id, c.idx)] = mre.tags_to_set(c.text, toks, crf.predict_single(mre.crf_features(toks)))

    res = {"data": {"n_docs": n_docs, "source_dirs": dirs,
                    "split_chunks": {k: len(v) for k, v in CH.items()}}, "wandb_mode": mode}
    sp_preds = None
    if not a.skip_spacy:
      sp_preds, sp_extra = run_spacy(CH, 3 if a.smoke else 30, log)
      res["spacy_ner"] = {"strict": mre.score(test, sp_preds), "relaxed": mre.score(test, sp_preds, relaxed=True), **sp_extra}
      print("spaCy strict micro-F1", round(res["spacy_ner"]["strict"]["micro_f1"], 4), flush=True)
    if crf_preds:
        res["crf"] = {"strict": mre.score(test, crf_preds)}
        if sp_preds is not None:
            res["bootstrap_spacy_minus_crf"] = mre.paired_bootstrap(test, sp_preds, crf_preds)

    if not a.skip_seq2seq:
        s2, raw, s2_extra, tch = run_seq2seq(CH, 1 if a.smoke else 15, log, smoke=a.smoke)
        res["flan_t5_small_seq2seq"] = {"strict": mre.score(tch, s2), "relaxed": mre.score(tch, s2, relaxed=True),
                                        "n_test_chunks": len(tch), **s2_extra}
        res["flan_t5_small_seq2seq"]["hallucination_rate"] = mre.hallucination_rate(
            tch, {k: [(t, x) for t, x in v] for k, v in s2.items()})
        if not a.smoke and sp_preds is not None:
            res["bootstrap_seq2seq_minus_spacy"] = mre.paired_bootstrap(test, s2, sp_preds)
        if not a.smoke and crf_preds:
            res["bootstrap_seq2seq_minus_crf"] = mre.paired_bootstrap(test, s2, crf_preds)
        json.dump({f"{k[0]}#{k[1]}": v for k, v in raw.items()}, open(f"{a.out}/raw_seq2seq.json", "w"))
        print("seq2seq strict micro-F1", round(res["flan_t5_small_seq2seq"]["strict"]["micro_f1"], 4), flush=True)

    summary = {k: round(v["strict"]["micro_f1"], 4) for k, v in res.items() if isinstance(v, dict) and "strict" in v}
    run.summary.update({f"test_strict_micro_f1/{k}": v for k, v in summary.items()})
    run.finish()
    json.dump(res, open(f"{a.out}/metrics_cpu.json", "w"), indent=1, default=float)
    print(json.dumps({"test_strict_micro_f1": summary,
                      **{k: v for k, v in res.items() if k.startswith("bootstrap")}}, indent=1, default=float))


if __name__ == "__main__":
    main()
