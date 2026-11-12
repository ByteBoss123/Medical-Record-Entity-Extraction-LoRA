"""Core utilities for medical-record entity extraction on MACCROBAT.

Data loading (brat standoff), document-level splits, chunking, scoring
(strict + relaxed set-based F1, hallucination rate), lexicon and CRF
baselines, LLM prompt/JSON helpers, and paired document-level bootstrap.
Everything here is CPU-only and dependency-light so it can be unit tested.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Entity types kept: the ones a personal-injury medical chronology needs.
TYPES = [
    "Disease_disorder",       # diagnoses / injuries
    "Sign_symptom",           # complaints, findings
    "Biological_structure",   # body region
    "Diagnostic_procedure",   # imaging, exams, tests
    "Therapeutic_procedure",  # surgery, therapy, treatment
    "Medication",
    "Date",
    "Duration",
]
TYPE_KEYS = {t: t.lower() for t in TYPES}  # JSON keys used in LLM output
KEY_TYPES = {v: k for k, v in TYPE_KEYS.items()}


@dataclass
class Span:
    type: str
    start: int
    end: int
    text: str
    discontinuous: bool = False


@dataclass
class Doc:
    doc_id: str
    text: str
    spans: list[Span] = field(default_factory=list)


@dataclass
class Chunk:
    doc_id: str
    idx: int
    start: int
    end: int
    text: str
    spans: list[Span]


# ----------------------------------------------------------------- loading
def parse_ann(ann_text: str, doc_text: str, keep=TYPES) -> list[Span]:
    spans = []
    for line in ann_text.splitlines():
        if not line.startswith("T"):
            continue  # skip events (E), relations (R), attributes (A), notes (#)
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        head, surface = parts[1], parts[2]
        typ, offs = head.split(" ", 1)
        if typ not in keep:
            continue
        frags = [tuple(map(int, f.split())) for f in offs.split(";")]
        start, end = min(f[0] for f in frags), max(f[1] for f in frags)
        disc = len(frags) > 1
        if not disc and doc_text[start:end] != surface:
            # offsets disagree with text: trust the surface string if locatable
            loc = doc_text.find(surface, max(0, start - 20))
            if loc == -1:
                continue
            start, end = loc, loc + len(surface)
        spans.append(Span(typ, start, end, surface, disc))
    spans.sort(key=lambda s: (s.start, -s.end))
    return spans


def load_brat_dir(root: str | Path) -> list[Doc]:
    root = Path(root)
    docs = []
    for txt in sorted(root.rglob("*.txt")):
        ann = txt.with_suffix(".ann")
        if not ann.exists():
            continue
        text = txt.read_text(encoding="utf-8")
        docs.append(Doc(txt.stem, text, parse_ann(ann.read_text(encoding="utf-8"), text)))
    # MACCROBAT2018 and 2020 releases can share PubMed IDs: keep one copy
    seen, uniq = set(), []
    for d in docs:
        if d.doc_id not in seen:
            seen.add(d.doc_id)
            uniq.append(d)
    return uniq


def split_docs(docs, seed=13, frac=(0.7, 0.1, 0.2)):
    ids = sorted(d.doc_id for d in docs)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    part = {i: "train" for i in ids[:a]} | {i: "val" for i in ids[a:b]} | {i: "test" for i in ids[b:]}
    out = {"train": [], "val": [], "test": []}
    for d in docs:
        out[part[d.doc_id]].append(d)
    return out


def chunk_doc(doc: Doc, max_lines=3, max_chars=900) -> list[Chunk]:
    """Group consecutive sentence lines; spans crossing a boundary are dropped
    from both chunks (counted by `cross_boundary_spans`)."""
    lines, pos = [], 0
    for ln in doc.text.split("\n"):
        lines.append((pos, pos + len(ln)))
        pos += len(ln) + 1
    groups, cur = [], []
    for s, e in lines:
        if e == s:
            continue
        if cur and (len(cur) >= max_lines or e - cur[0][0] > max_chars):
            groups.append(cur)
            cur = []
        cur.append((s, e))
    if cur:
        groups.append(cur)
    chunks = []
    for i, g in enumerate(groups):
        s, e = g[0][0], g[-1][1]
        inside = [Span(sp.type, sp.start - s, sp.end - s, sp.text, sp.discontinuous)
                  for sp in doc.spans if sp.start >= s and sp.end <= e]
        chunks.append(Chunk(doc.doc_id, i, s, e, doc.text[s:e], inside))
    return chunks


def cross_boundary_spans(doc: Doc, chunks: list[Chunk]) -> int:
    return len(doc.spans) - sum(len(c.spans) for c in chunks)


# ----------------------------------------------------------------- scoring
_ws = re.compile(r"\s+")


def norm(s: str) -> str:
    s = _ws.sub(" ", s.lower()).strip()
    return s.strip(" .,;:()[]\"'")


def gold_set(chunk: Chunk) -> set[tuple[str, str]]:
    return {(sp.type, norm(sp.text)) for sp in chunk.spans if norm(sp.text)}


def _toks(s):
    return set(re.findall(r"\w+", s))


def relaxed_match(a: str, b: str) -> bool:
    if a == b or a in b or b in a:
        return True
    ta, tb = _toks(a), _toks(b)
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= 0.5


def match_counts(pred: set, gold: set, relaxed=False):
    """Return tp, fp, fn per type. Relaxed uses greedy one-to-one matching."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for t in TYPES:
        p = sorted(x for (ty, x) in pred if ty == t)
        g = sorted(x for (ty, x) in gold if ty == t)
        if not relaxed:
            k = len(set(p) & set(g))
            tp[t] += k
            fp[t] += len(p) - k
            fn[t] += len(g) - k
            continue
        used = set()
        for x in p:
            hit = next((j for j, y in enumerate(g) if j not in used and relaxed_match(x, y)), None)
            if hit is None:
                fp[t] += 1
            else:
                used.add(hit)
                tp[t] += 1
        fn[t] += len(g) - len(used)
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score(chunks: list[Chunk], preds: dict, relaxed=False):
    """preds: {(doc_id, idx): set[(type, normtext)]}"""
    TP, FP, FN = Counter(), Counter(), Counter()
    for c in chunks:
        tp, fp, fn = match_counts(preds.get((c.doc_id, c.idx), set()), gold_set(c), relaxed)
        TP += tp; FP += fp; FN += fn
    per = {t: dict(zip(("precision", "recall", "f1"), prf(TP[t], FP[t], FN[t])),
                   support=TP[t] + FN[t]) for t in TYPES}
    p, r, f = prf(sum(TP.values()), sum(FP.values()), sum(FN.values()))
    macro = sum(per[t]["f1"] for t in TYPES) / len(TYPES)
    return {"micro_precision": p, "micro_recall": r, "micro_f1": f, "macro_f1": macro, "per_type": per}


def doc_counts(chunks, preds, relaxed=False):
    """Per-document summed tp/fp/fn (all types) for bootstrap."""
    out = defaultdict(lambda: [0, 0, 0])
    for c in chunks:
        tp, fp, fn = match_counts(preds.get((c.doc_id, c.idx), set()), gold_set(c), relaxed)
        v = out[c.doc_id]
        v[0] += sum(tp.values()); v[1] += sum(fp.values()); v[2] += sum(fn.values())
    return dict(out)


def paired_bootstrap(chunks, preds_a, preds_b, n=2000, seed=0, relaxed=False):
    """Doc-level paired bootstrap of micro-F1(a) - micro-F1(b)."""
    da, db = doc_counts(chunks, preds_a, relaxed), doc_counts(chunks, preds_b, relaxed)
    ids = sorted(set(da) | set(db))
    rng = random.Random(seed)

    def f1(d, sample):
        tp = sum(d.get(i, [0, 0, 0])[0] for i in sample)
        fp = sum(d.get(i, [0, 0, 0])[1] for i in sample)
        fn = sum(d.get(i, [0, 0, 0])[2] for i in sample)
        return prf(tp, fp, fn)[2]

    obs = f1(da, ids) - f1(db, ids)
    diffs = sorted(f1(da, s) - f1(db, s) for s in
                   ([rng.choice(ids) for _ in ids] for _ in range(n)))
    lo, hi = diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]
    return {"diff": obs, "ci95": [lo, hi], "n_docs": len(ids), "n_boot": n}


def hallucination_rate(chunks, raw_preds):
    """Share of predicted strings not found (normalized) in the source chunk."""
    by = {(c.doc_id, c.idx): norm(c.text) for c in chunks}
    tot = bad = 0
    for k, items in raw_preds.items():
        src = by.get(k, "")
        for _, s in items:
            tot += 1
            bad += norm(s) not in src
    return bad / tot if tot else 0.0


# ------------------------------------------------------- lexicon baseline
def build_lexicon(chunks, min_len=3):
    votes = defaultdict(Counter)
    for c in chunks:
        for sp in c.spans:
            n = norm(sp.text)
            if len(n) >= min_len:
                votes[n][sp.type] += 1
    return {k: v.most_common(1)[0][0] for k, v in votes.items()}


def lexicon_predict(chunk, lex):
    text = chunk.text.lower()
    taken = [False] * len(text)
    out = set()
    for term in sorted(lex, key=len, reverse=True):
        for m in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text):
            if not any(taken[m.start():m.end()]):
                for i in range(m.start(), m.end()):
                    taken[i] = True
                out.add((lex[term], term))
    return out


# ----------------------------------------------------------- CRF baseline
_tok = re.compile(r"\w+|[^\w\s]")


def tokenize(text):
    return [(m.group(), m.start(), m.end()) for m in _tok.finditer(text)]


def bio_labels(chunk):
    """BIO tags from char spans; longer spans win overlaps. Returns tags and
    the number of gold spans that could not be encoded (nested/overlapping)."""
    toks = tokenize(chunk.text)
    tags = ["O"] * len(toks)
    dropped = 0
    for sp in sorted(chunk.spans, key=lambda s: -(s.end - s.start)):
        idx = [i for i, (_, s, e) in enumerate(toks) if s >= sp.start and e <= sp.end]
        if not idx or any(tags[i] != "O" for i in idx):
            dropped += 1
            continue
        tags[idx[0]] = "B-" + sp.type
        for i in idx[1:]:
            tags[i] = "I-" + sp.type
    return toks, tags, dropped


def tags_to_set(text, toks, tags):
    out, cur = set(), None
    for (w, s, e), t in zip(toks, tags):
        if t.startswith("B-") or (t.startswith("I-") and (cur is None or cur[0] != t[2:])):
            if cur:
                out.add((cur[0], norm(text[cur[1]:cur[2]])))
            cur = [t[2:], s, e]
        elif t.startswith("I-"):
            cur[2] = e
        else:
            if cur:
                out.add((cur[0], norm(text[cur[1]:cur[2]])))
            cur = None
    if cur:
        out.add((cur[0], norm(text[cur[1]:cur[2]])))
    return {x for x in out if x[1]}


def _wfeat(w):
    return {"w": w.lower(), "suf3": w[-3:].lower(), "pre3": w[:3].lower(),
            "title": w.istitle(), "upper": w.isupper(), "digit": w.isdigit(),
            "hasdigit": any(ch.isdigit() for ch in w), "len": min(len(w), 12)}


def crf_features(toks):
    feats = []
    for i, (w, _, _) in enumerate(toks):
        f = {"bias": 1.0, **_wfeat(w)}
        for d in (-2, -1, 1, 2):
            j = i + d
            if 0 <= j < len(toks):
                for k, v in _wfeat(toks[j][0]).items():
                    if k in ("w", "suf3", "title", "digit"):
                        f[f"{d}:{k}"] = v
            else:
                f[f"{d}:pad"] = True
        feats.append(f)
    return feats


# ---------------------------------------------------------- LLM helpers
SYSTEM = ("You extract medical facts from clinical text for personal-injury case review. "
          "Copy each mention exactly as written in the text. Do not paraphrase, infer, or add "
          "anything not in the text. Return only JSON.")


def instruction(text):
    keys = ", ".join(TYPE_KEYS[t] for t in TYPES)
    return (f"Extract every mention of these types: {keys}.\n"
            "Return a JSON object with exactly those keys; each value is a list of exact "
            "substrings from the text (empty list if none).\n\nText:\n" + text)


def target_json(chunk):
    out = {TYPE_KEYS[t]: [] for t in TYPES}
    for sp in sorted(chunk.spans, key=lambda s: s.start):
        lst = out[TYPE_KEYS[sp.type]]
        if sp.text not in lst:
            lst.append(sp.text)
    return json.dumps(out, ensure_ascii=False)


def parse_llm_json(s):
    """Return (set of (type, normtext), list of (type, raw), valid_json)."""
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return set(), [], False
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return set(), [], False
    if not isinstance(obj, dict):
        return set(), [], False
    raw = []
    for k, v in obj.items():
        t = KEY_TYPES.get(str(k).lower())
        if t is None or not isinstance(v, list):
            continue
        raw += [(t, x) for x in v if isinstance(x, str) and x.strip()]
    return {(t, norm(x)) for t, x in raw if norm(x)}, raw, True
