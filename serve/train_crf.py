"""Train the CRF extractor on the MACCROBAT train split and save it for serving.

Usage: python serve/train_crf.py [--data DIR] [--out serve/model]
Without --data, downloads MACCROBAT from figshare (article 9764942).
"""
import argparse
import glob
import io
import json
import os
import pickle
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mre  # noqa: E402
import sklearn_crfsuite  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (research)"}


def _get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()


def _extract(blob, dest):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for n in z.namelist():
            if n.endswith(".zip"):
                _extract(z.read(n), os.path.join(dest, Path(n).stem))
            elif not n.endswith("/"):
                p = os.path.join(dest, n)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, "wb").write(z.read(n))


def download(dest="data"):
    api = json.loads(_get("https://api.figshare.com/v2/articles/9764942"))
    for f in api["files"]:
        if f["name"].endswith(".zip"):
            _extract(_get(f["download_url"]), os.path.join(dest, Path(f["name"]).stem))
    dirs = sorted({os.path.dirname(p) for p in glob.glob(f"{dest}/**/*.ann", recursive=True)})
    return [d for d in dirs if "2020" in d] or dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--out", default=str(ROOT / "serve" / "model"))
    a = ap.parse_args()
    dirs = [a.data] if a.data else download()
    docs, seen = [], set()
    for d in dirs:
        for doc in mre.load_brat_dir(d):
            if doc.doc_id not in seen:
                seen.add(doc.doc_id)
                docs.append(doc)
    sp = mre.split_docs(docs, seed=13)
    ch = {k: [c for d in v for c in mre.chunk_doc(d)] for k, v in sp.items()}
    X, Y = [], []
    for c in ch["train"]:
        toks, tags, _ = mre.bio_labels(c)
        X.append(mre.crf_features(toks))
        Y.append(tags)
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=200,
                               all_possible_transitions=True)
    crf.fit(X, Y)
    preds = {}
    for c in ch["test"]:
        toks = mre.tokenize(c.text)
        preds[(c.doc_id, c.idx)] = mre.tags_to_set(c.text, toks, crf.predict_single(mre.crf_features(toks)))
    m = mre.score(ch["test"], preds)
    os.makedirs(a.out, exist_ok=True)
    pickle.dump(crf, open(os.path.join(a.out, "crf.pkl"), "wb"))
    meta = {"n_docs": len(docs), "split_docs": {k: len(v) for k, v in sp.items()},
            "test_strict_micro_f1": m["micro_f1"], "test_strict_macro_f1": m["macro_f1"],
            "source_dirs": dirs}
    json.dump(meta, open(os.path.join(a.out, "train_meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
