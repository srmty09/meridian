import os

MODEL_COST_PER_1K_TOKENS = {
    "small": float(os.getenv("COST_SMALL", "0.00010")),
    "medium": float(os.getenv("COST_MEDIUM", "0.00030")),
    "large": float(os.getenv("COST_LARGE", "0.00090")),
}

BASELINE_MODEL = "large"
COST_LABEL = "Estimated Cost"


def estimate_cost(model_name: str, tokens: int) -> float:
    price = MODEL_COST_PER_1K_TOKENS[model_name]
    return price * tokens / 1000.0


def baseline_cost(tokens: int) -> float:
    return estimate_cost(BASELINE_MODEL, tokens)


def savings(baseline_total: float, actual_total: float):
    cost_saved = baseline_total - actual_total
    pct = (cost_saved / baseline_total * 100.0) if baseline_total > 0 else 0.0
    return cost_saved, pct
