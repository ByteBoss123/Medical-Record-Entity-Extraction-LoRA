"""Unit tests on a tiny hand-written brat fixture (format check only; not data)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mre  # noqa: E402

TEXT = ("A 45-year-old man presented with neck pain after a collision.\n"
        "CT of the cervical spine showed a C5 fracture.\n"
        "He underwent anterior fusion and received ibuprofen for 6 weeks.\n")


def _off(s, nth=0):
    i = -1
    for _ in range(nth + 1):
        i = TEXT.index(s, i + 1)
    return i, i + len(s)


def ann():
    rows = []
    for k, (t, s) in enumerate([
        ("Sign_symptom", "pain"), ("Biological_structure", "neck"),
        ("Diagnostic_procedure", "CT"), ("Biological_structure", "cervical spine"),
        ("Disease_disorder", "fracture"), ("Therapeutic_procedure", "anterior fusion"),
        ("Medication", "ibuprofen"), ("Duration", "6 weeks"), ("Age", "45-year-old"),
    ], start=1):
        a, b = _off(s)
        rows.append(f"T{k}\t{t} {a} {b}\t{s}")
    a1, b1 = _off("C5"); a2, b2 = _off("fracture")
    rows.append(f"T20\tDisease_disorder {a1} {b1};{a2} {b2}\tC5 fracture")
    rows += ["E1\tSign_symptom:T1", "R1\tMODIFY Arg1:T2 Arg2:E1", "A1\tPolarity E1 NEG", "#1\tAnnotatorNotes T1\tx"]
    return "\n".join(rows)


def doc(tmp_path):
    (tmp_path / "1.txt").write_text(TEXT)
    (tmp_path / "1.ann").write_text(ann())
    return mre.load_brat_dir(tmp_path)[0]


def test_parse(tmp_path):
    d = doc(tmp_path)
    types = [s.type for s in d.spans]
    assert "Age" not in types and len(d.spans) == 9
    disc = [s for s in d.spans if s.discontinuous]
    assert len(disc) == 1 and disc[0].text == "C5 fracture"


def test_chunk_and_scoring(tmp_path):
    d = doc(tmp_path)
    ch = mre.chunk_doc(d, max_lines=2)
    assert len(ch) == 2 and mre.cross_boundary_spans(d, ch) == 0
    gold = {(c.doc_id, c.idx): mre.gold_set(c) for c in ch}
    s = mre.score(ch, gold)
    assert s["micro_f1"] == 1.0
    preds = {k: set(v) for k, v in gold.items()}
    preds[(d.doc_id, 0)].add(("Medication", "aspirin"))
    preds[(d.doc_id, 1)].discard(("Medication", "ibuprofen"))
    preds[(d.doc_id, 1)].add(("Medication", "ibuprofen tablets"))
    strict, relax = mre.score(ch, preds), mre.score(ch, preds, relaxed=True)
    assert strict["per_type"]["Medication"]["f1"] == 0.0
    assert relax["per_type"]["Medication"]["precision"] == 0.5


def test_lexicon_and_bio(tmp_path):
    d = doc(tmp_path)
    ch = mre.chunk_doc(d, max_lines=3)
    lex = mre.build_lexicon(ch)
    p = mre.lexicon_predict(ch[0], lex)
    assert ("Medication", "ibuprofen") in p and ("Disease_disorder", "c5 fracture") in p
    toks, tags, dropped = mre.bio_labels(ch[0])
    # nested "fracture" inside discontinuous "C5 fracture" cannot be encoded
    assert dropped == 1
    back = mre.tags_to_set(ch[0].text, toks, tags)
    assert ("Therapeutic_procedure", "anterior fusion") in back


def test_llm_json(tmp_path):
    d = doc(tmp_path)
    ch = mre.chunk_doc(d)
    tgt = mre.target_json(ch[0])
    s, raw, ok = mre.parse_llm_json("Sure:\n" + tgt + "\nDone")
    assert ok and s == mre.gold_set(ch[0])
    assert mre.parse_llm_json("{bad json")[2] is False
    preds = {(d.doc_id, 0): [("Medication", "ibuprofen"), ("Medication", "morphine")]}
    assert mre.hallucination_rate(ch, preds) == 0.5


def test_bootstrap(tmp_path):
    d = doc(tmp_path)
    ch = mre.chunk_doc(d)
    gold = {(c.doc_id, c.idx): mre.gold_set(c) for c in ch}
    r = mre.paired_bootstrap(ch, gold, {}, n=200)
    assert r["diff"] == 1.0 and r["ci95"][0] == 1.0


def test_ocr_roundtrip():
    import ocr
    assert ocr.cer("abc def", "abc  def") == 0.0 and ocr.cer("abcd", "abxd") == 0.25
    _, lv, out, c = ocr.ocr_job((0, TEXT, "clean", 0))
    assert lv == "clean" and c < 0.02 and "cervical spine" in out
