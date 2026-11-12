"""FastAPI service: POST /extract {"text": ...} -> entities with char offsets."""
import json
import os
import pickle
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mre  # noqa: E402

MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "serve" / "model"))
CRF = pickle.load(open(MODEL_DIR / "crf.pkl", "rb"))
META = json.load(open(MODEL_DIR / "train_meta.json"))
MAX_CHARS = 20000
app = FastAPI(title="Medical-record entity extraction")


class Req(BaseModel):
    text: str


def spans(text, toks, tags):
    out, cur = [], None
    for (_, s, e), t in zip(toks, tags):
        if t.startswith("B-") or (t.startswith("I-") and (cur is None or cur["type"] != t[2:])):
            if cur:
                out.append(cur)
            cur = {"type": t[2:], "start": s, "end": e}
        elif t.startswith("I-"):
            cur["end"] = e
        else:
            if cur:
                out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    for x in out:
        x["text"] = text[x["start"]:x["end"]]
    return out


@app.get("/health")
def health():
    return {"status": "ok", "model": "crf", "test_strict_micro_f1": META["test_strict_micro_f1"]}


@app.post("/extract")
def extract(r: Req):
    if not r.text.strip():
        raise HTTPException(422, "empty text")
    if len(r.text) > MAX_CHARS:
        raise HTTPException(413, f"text longer than {MAX_CHARS} chars")
    t0 = time.perf_counter()
    ents = []
    for line_start, line in _lines(r.text):
        toks = mre.tokenize(line)
        if toks:
            for x in spans(line, toks, CRF.predict_single(mre.crf_features(toks))):
                x["start"] += line_start
                x["end"] += line_start
                ents.append(x)
    return {"entities": ents, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


def _lines(text):
    pos = 0
    for ln in text.split("\n"):
        yield pos, ln
        pos += len(ln) + 1
