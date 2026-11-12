"""Simulated-scan OCR robustness: render real case-report text as a page
image, apply scanner-style degradation, OCR it with Tesseract, measure CER.

The degradation is synthetic (applied to real text); these are not real
scanned records. Levels were calibrated on a hand-written sample (target
roughly 3-5% and ~10% CER) before any MACCROBAT result existed, then frozen.
"""
from __future__ import annotations

import io
import random
import textwrap

from PIL import Image, ImageDraw, ImageFilter, ImageFont

LEVELS = {
    "clean":  dict(blur=0.0, noise=0.00, rotate=0.0, jpeg=95, scale=1.0),
    "mild":   dict(blur=0.4, noise=0.010, rotate=0.5, jpeg=70, scale=0.9),
    "heavy":  dict(blur=0.6, noise=0.018, rotate=1.0, jpeg=45, scale=0.8),
}
_FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"]


def _font(size=22):
    for p in _FONTS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def render(text: str, level: str, seed: int = 0, width_chars: int = 80) -> Image.Image:
    cfg, rng = LEVELS[level], random.Random(seed)
    font = _font()
    lines = []
    for para in text.split("\n"):
        lines += textwrap.wrap(para, width_chars) or [""]
    lh = 30
    img = Image.new("L", (1000, 60 + lh * len(lines)), 255)
    d = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        d.text((40, 30 + i * lh), ln, fill=0, font=font)
    if cfg["rotate"]:
        img = img.rotate(rng.uniform(-cfg["rotate"], cfg["rotate"]), expand=True, fillcolor=255)
    if cfg["scale"] != 1.0:
        w, h = img.size
        img = img.resize((int(w * cfg["scale"]), int(h * cfg["scale"])), Image.BILINEAR)
    if cfg["blur"]:
        img = img.filter(ImageFilter.GaussianBlur(cfg["blur"]))
    if cfg["noise"]:
        px = img.load()
        w, h = img.size
        for _ in range(int(cfg["noise"] * w * h)):
            x, y = rng.randrange(w), rng.randrange(h)
            px[x, y] = 0 if rng.random() < 0.5 else 255
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=cfg["jpeg"])
    return Image.open(io.BytesIO(buf.getvalue()))


def ocr(img: Image.Image) -> str:
    import pytesseract
    return pytesseract.image_to_string(img, config="--psm 6").strip()


def _lev(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """Character error rate on whitespace-collapsed text."""
    r, h = " ".join(ref.split()), " ".join(hyp.split())
    return _lev(r, h) / max(1, len(r))


def ocr_job(args):
    """(key, text, level, seed) -> (key, level, ocr_text, cer). Pool-friendly."""
    key, text, level, seed = args
    out = ocr(render(text, level, seed))
    return key, level, out, cer(text, out)
