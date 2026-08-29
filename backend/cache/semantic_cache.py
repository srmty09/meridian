from cache import verifier
from cache.retriever import Retriever

_retriever = Retriever()


def add(prompt: str, response: str, model: str, tokens: int):
    _retriever.add(prompt, response, model, tokens)


def lookup(prompt: str):
    candidates = _retriever.retrieve(prompt)
    considered = []
    for cand in candidates:
        equivalent, s = verifier.is_equivalent(prompt, cand["prompt"])
        considered.append({"prompt": cand["prompt"], "score": s, "equivalent": equivalent})
        if equivalent:
            return {
                "response": cand["response"],
                "model": cand["model"],
                "tokens": cand["tokens"],
                "matched_prompt": cand["prompt"],
                "score": s,
                "considered": considered,
            }
    return None


def size():
    return len(_retriever)


ADVERSARIAL_PAIRS = [
    ("Is aspirin safe with warfarin?", "Is aspirin safe without warfarin?", "negation"),
    ("What is the population of India?", "What was the population of India in 1950?", "different date"),
    ("Who is the CEO of Apple?", "Who is the CEO of Microsoft?", "entity swap"),
    ("How do I enable dark mode on iOS?", "How do I disable dark mode on iOS?", "negation"),
    ("What is the boiling point of water at sea level?",
     "What is the boiling point of water at 4000 metres?", "different condition"),
    ("Convert 10 kilometres to miles.", "Convert 100 kilometres to miles.", "different quantity"),
    ("What year did World War II end?", "What year did World War I end?", "entity swap"),
    ("Is it safe to take ibuprofen while pregnant?",
     "Is it safe to take ibuprofen while breastfeeding?", "different condition"),
]


def run_adversarial_test():
    local = Retriever()
    results = []
    false_hits = 0

    for seeded, query, reason in ADVERSARIAL_PAIRS:
        local.add(seeded, "<cached answer>", "small", 42)
        hit = None
        for cand in local.retrieve(query):
            equivalent, s = verifier.is_equivalent(query, cand["prompt"])
            if equivalent:
                hit = {"matched": cand["prompt"], "score": s}
                break
        is_false_hit = hit is not None
        if is_false_hit:
            false_hits += 1
        results.append({
            "seeded_prompt": seeded,
            "adversarial_query": query,
            "reason": reason,
            "false_hit": is_false_hit,
            "verifier_score": hit["score"] if hit else None,
        })

    return {
        "pairs": len(ADVERSARIAL_PAIRS),
        "false_hits": false_hits,
        "false_hit_rate": false_hits / len(ADVERSARIAL_PAIRS),
        "semantic_threshold": verifier.SEMANTIC_THRESHOLD,
        "verifier_model": verifier.VERIFIER_MODEL,
        "results": results,
    }
