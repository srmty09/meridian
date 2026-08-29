import hashlib
import json
import os
import re
import time

try:
    import redis
except ImportError:
    redis = None

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", str(7 * 24 * 3600)))
KEY_PREFIX = "meridian:exact:v1:"

_backend = None
_ws_re = re.compile(r"\s+")


class _MemoryBackend:
    kind = "memory"

    def __init__(self, reason):
        self._d = {}
        print(f"[redis_cache] using in-memory dict ({reason}) - data lost on restart")

    def get(self, key):
        return self._d.get(key)

    def set(self, key, value, ex=None):
        self._d[key] = value

    def size(self):
        return len(self._d)


class _RedisBackend:
    kind = "redis"

    def __init__(self, client):
        self._c = client

    def get(self, key):
        return self._c.get(key)

    def set(self, key, value, ex=None):
        self._c.set(key, value, ex=ex or None)

    def size(self):
        return sum(1 for _ in self._c.scan_iter(match=KEY_PREFIX + "*", count=500))


def _connect():
    global _backend
    if redis is not None:
        try:
            client = redis.from_url(REDIS_URL, decode_responses=True,
                                    socket_connect_timeout=2, socket_timeout=2)
            client.ping()
            print(f"[redis_cache] connected to {REDIS_URL}  (ttl={CACHE_TTL_SECONDS}s)")
            _backend = _RedisBackend(client)
            return _backend
        except Exception as e:
            _backend = _MemoryBackend(f"connect failed: {e}")
            return _backend
    _backend = _MemoryBackend("redis-py not installed")
    return _backend


def _get_backend():
    return _backend if _backend is not None else _connect()


def _demote(err):
    global _backend
    if not isinstance(_backend, _MemoryBackend):
        _backend = _MemoryBackend(f"runtime error: {err}")


def _norm(prompt: str) -> str:
    return _ws_re.sub(" ", prompt).strip()


def _key(prompt: str) -> str:
    return KEY_PREFIX + hashlib.sha256(_norm(prompt).encode("utf-8")).hexdigest()


def get(prompt: str):
    try:
        raw = _get_backend().get(_key(prompt))
    except Exception as e:
        _demote(e)
        raw = _get_backend().get(_key(prompt))
    return json.loads(raw) if raw else None


def put(prompt: str, response: str, model: str, tokens: int):
    payload = json.dumps({
        "prompt": prompt, "response": response, "model": model,
        "tokens": tokens, "created_at": time.time(),
    })
    try:
        _get_backend().set(_key(prompt), payload, ex=CACHE_TTL_SECONDS)
    except Exception as e:
        _demote(e)
        _get_backend().set(_key(prompt), payload, ex=CACHE_TTL_SECONDS)


def stats():
    b = _get_backend()
    try:
        return {"backend": b.kind, "entries": b.size()}
    except Exception as e:
        _demote(e)
        return {"backend": "memory", "entries": _get_backend().size()}
