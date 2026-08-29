import os
import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROUTER_MODEL = os.getenv("ROUTER_MODEL", "srmty/meridian")
ROUTER_THRESHOLD = float(os.getenv("ROUTER_THRESHOLD", "0.75"))
BLEND_BELOW = float(os.getenv("ROUTER_BLEND_BELOW", "0.55"))

ID2LABEL = {0: "small", 1: "medium", 2: "large"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
NEXT_SIZE = {"small": "medium", "medium": "large", "large": "large"}

_tokenizer = None
_model = None
_load_error = None


def _load():
    global _tokenizer, _model, _load_error
    if _model is not None or _load_error is not None:
        return
    try:
        _tokenizer = AutoTokenizer.from_pretrained(ROUTER_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(ROUTER_MODEL)
        _model.eval()
        print(f"[router] loaded {ROUTER_MODEL}")
    except Exception as e:
        _load_error = e
        print(f"[router] could not load {ROUTER_MODEL}: {e}\n"
              f"[router] falling back to a keyword heuristic")


_MATH_RE = re.compile(r"^[\s\d.,+\-*/^%()x×÷=]+\??$")
_HARD = ("prove", "derive", "analyze", "analyse", "explain why", "design ", "architect",
         "implement", "optimize", "optimise", "complexity", "algorithm", "compare",
         "evaluate", "essay", "step by step", "in detail", "pros and cons", "trade-off",
         "tradeoff", "reason through", "walk me through", "from scratch", "critique")
_MEDIUM = ("summarize", "summarise", "explain", "describe", "translate", "how do i",
           "how does", "how can i", "what are", "tell me about", "rewrite", "outline",
           "give an example", "write a", "generate", "draft", "debug", "refactor",
           "recommend", "suggest", "difference between", "benefits of", "list ")
_EASY = ("capital of", "who is ", "who was ", "when did", "when was", "how many",
         "what is the ", "define ", "spell ", "convert ", " abbreviation", " * ",
         " + ", " times ", "square root of")


def _heuristic(prompt: str):
    p = prompt.lower().strip()
    n = len(prompt)

    if any(k in p for k in _HARD) or n > 320:
        return [0.10, 0.25, 0.65]
    if any(k in p for k in _MEDIUM) or n > 140:
        return [0.20, 0.58, 0.22]
    if _MATH_RE.match(p) or any(k in p for k in _EASY) or n < 40:
        return [0.72, 0.20, 0.08]
    return [0.20, 0.58, 0.22]


def route(prompt: str):
    _load()

    heur = _heuristic(prompt)
    if _model is not None:
        enc = _tokenizer(prompt, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = _model(**enc).logits[0]
        model_probs = torch.softmax(logits, dim=-1).tolist()
    else:
        model_probs = heur

    model_conf = max(model_probs)
    unsure = _model is not None and BLEND_BELOW > 0 and model_conf < BLEND_BELOW

    if unsure:
        probs = heur
        chosen = predicted = ID2LABEL[max(range(3), key=lambda i: heur[i])]
        escalated = False
    else:
        probs = model_probs
        predicted = ID2LABEL[max(range(3), key=lambda i: probs[i])]
        chosen, escalated = predicted, False
        if max(probs) < ROUTER_THRESHOLD:
            chosen = NEXT_SIZE[predicted]
            escalated = chosen != predicted

    return {
        "model": chosen,
        "predicted": predicted,
        "confidence": max(probs),
        "probs": {ID2LABEL[i]: round(probs[i], 3) for i in range(3)},
        "model_probs": {ID2LABEL[i]: round(model_probs[i], 3) for i in range(3)},
        "blended_with_heuristic": unsure,
        "escalated": escalated,
        "using_fallback": _model is None,
    }
