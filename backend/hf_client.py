import os
import re

from openai import OpenAI

MODELS = {
    "small": os.getenv("MODEL_SMALL", "Qwen/Qwen3-0.6B:featherless-ai"),
    "medium": os.getenv("MODEL_MEDIUM", "Qwen/Qwen3-1.7B:featherless-ai"),
    "large": os.getenv("MODEL_LARGE", "Qwen/Qwen3-8B:featherless-ai"),
}

BASE_URL = "https://router.huggingface.co/v1"

_client = None

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPEN_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)


def _get_client():
    global _client
    if _client is None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is not set. Copy .env.example to .env and fill it in.")
        _client = OpenAI(base_url=BASE_URL, api_key=token)
    return _client


def _strip_thinking(text: str) -> str:
    t = _THINK_RE.sub("", text).strip()
    if t:
        return t
    if "</think>" in text:
        tail = text.rsplit("</think>", 1)[-1].strip()
        if tail:
            return tail
    return _OPEN_THINK_RE.sub("", text).replace("<think>", "").replace("</think>", "").strip()


def _one_call(client, model_id, prompt, max_tokens, temperature):
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are a helpful, concise assistant. /no_think"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = _strip_thinking(resp.choices[0].message.content or "")
    usage = resp.usage
    tokens = usage.total_tokens if (usage and usage.total_tokens) else (len(prompt) + len(text)) // 4
    return text, tokens


def generate(prompt: str, model_name: str, max_tokens: int = 512, temperature: float = 0.3):
    if model_name not in MODELS:
        raise ValueError(f"unknown model size {model_name!r}, expected one of {list(MODELS)}")

    client = _get_client()
    text, tokens = _one_call(client, MODELS[model_name], prompt, max_tokens, temperature)
    if not text:
        text, tokens = _one_call(client, MODELS[model_name], prompt, max_tokens * 2, temperature)
    return text or "(the model returned an empty response)", tokens
