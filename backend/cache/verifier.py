import math
import os
import re

from sentence_transformers import CrossEncoder

SIM_MODEL = os.getenv("VERIFIER_SIM_MODEL", "cross-encoder/stsb-distilroberta-base")
NLI_MODEL = os.getenv("VERIFIER_NLI_MODEL", "cross-encoder/nli-deberta-v3-xsmall")
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.50"))

VERIFIER_MODEL = f"{SIM_MODEL} + {NLI_MODEL}"

_sim = None
_nli = None
_nli_labels = None
_nli_failed = False

_NUM_RE = re.compile(r"\b\d[\d,]*\b")


def _sim_model():
    global _sim
    if _sim is None:
        _sim = CrossEncoder(SIM_MODEL)
    return _sim


def _nli_model():
    global _nli, _nli_labels, _nli_failed
    if _nli is None and not _nli_failed:
        try:
            _nli = CrossEncoder(NLI_MODEL)
            _nli_labels = {int(k): v.lower() for k, v in _nli.model.config.id2label.items()}
        except Exception as e:
            _nli_failed = True
            print(f"[verifier] NLI model unavailable ({e}) - similarity-only, guards off")
    return _nli


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _similarity(a: str, b: str) -> float:
    vals = _sim_model().predict([(a, b), (b, a)])
    s = min(float(v) for v in vals)
    return s if 0.0 <= s <= 1.0 else _sigmoid(s)


def _contradicts(a: str, b: str) -> bool:
    model = _nli_model()
    if model is None:
        return False
    import numpy as np
    for pair in ((a, b), (b, a)):
        logits = np.asarray(model.predict([pair])[0], dtype=float)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        if _nli_labels[int(probs.argmax())] == "contradiction":
            return True
    return False


def _number_mismatch(a: str, b: str) -> bool:
    na = set(_NUM_RE.findall(a.replace(",", "")))
    nb = set(_NUM_RE.findall(b.replace(",", "")))
    return na != nb


def score(a: str, b: str) -> float:
    if _number_mismatch(a, b) or _contradicts(a, b):
        return 0.0
    return _similarity(a, b)


def is_equivalent(a: str, b: str):
    s = score(a, b)
    return s >= SEMANTIC_THRESHOLD, s
