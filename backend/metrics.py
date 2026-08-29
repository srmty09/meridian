import threading

from cost import savings

_lock = threading.Lock()

_state = {
    "total_requests": 0,
    "exact_cache_hits": 0,
    "exact_cache_misses": 0,
    "semantic_cache_hits": 0,
    "semantic_cache_misses": 0,
    "semantic_false_hits": 0,
    "routing": {"small": 0, "medium": 0, "large": 0},
    "estimated_cost": 0.0,
    "baseline_cost": 0.0,
    "router_confidences": [],
    "latencies_ms": [],
}


def record_request(*, cache_type, model, router_confidence, estimated_cost, baseline_cost, latency_ms):
    with _lock:
        _state["total_requests"] += 1

        if cache_type == "exact":
            _state["exact_cache_hits"] += 1
        elif cache_type == "semantic":
            _state["exact_cache_misses"] += 1
            _state["semantic_cache_hits"] += 1
        else:
            _state["exact_cache_misses"] += 1
            _state["semantic_cache_misses"] += 1
            if model in _state["routing"]:
                _state["routing"][model] += 1

        _state["estimated_cost"] += estimated_cost
        _state["baseline_cost"] += baseline_cost
        if router_confidence is not None:
            _state["router_confidences"].append(router_confidence)
        _state["latencies_ms"].append(latency_ms)


def record_false_hit():
    with _lock:
        _state["semantic_false_hits"] += 1


def _avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def snapshot():
    with _lock:
        total = _state["total_requests"]
        exact = _state["exact_cache_hits"]
        semantic = _state["semantic_cache_hits"]
        misses = _state["semantic_cache_misses"]
        cost_saved, pct = savings(_state["baseline_cost"], _state["estimated_cost"])

        return {
            "total_requests": total,
            "cache_hit_rate": (exact + semantic) / total if total else 0.0,
            "exact_cache_hits": exact,
            "exact_cache_misses": _state["exact_cache_misses"],
            "semantic_cache_hits": semantic,
            "semantic_cache_misses": misses,
            "semantic_false_hits": _state["semantic_false_hits"],
            "cache_misses": misses,
            "routing_distribution": dict(_state["routing"]),
            "estimated_cost": _state["estimated_cost"],
            "baseline_cost": _state["baseline_cost"],
            "cost_saved": cost_saved,
            "saving_percentage": pct,
            "avg_router_confidence": _avg(_state["router_confidences"]),
            "avg_latency_ms": _avg(_state["latencies_ms"]),
        }


def reset():
    with _lock:
        _state.update({
            "total_requests": 0,
            "exact_cache_hits": 0,
            "exact_cache_misses": 0,
            "semantic_cache_hits": 0,
            "semantic_cache_misses": 0,
            "semantic_false_hits": 0,
            "routing": {"small": 0, "medium": 0, "large": 0},
            "estimated_cost": 0.0,
            "baseline_cost": 0.0,
            "router_confidences": [],
            "latencies_ms": [],
        })
