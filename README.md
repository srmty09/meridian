# Meridian

**A learned LLM router with a verified semantic cache.**

Most requests to a large model don't need a large model. Meridian learns, per
prompt, the *cheapest model that is still good enough*, and it reuses past
answers - but only after a cross-encoder confirms the new prompt is really asking
the same thing.

```
User prompt
   |
   v
FastAPI
   |
   +--> Redis exact cache ........... SHA256(prompt) hit? -> return
   |
   +--> Semantic cache
   |       TF-IDF / BM25 / dense retrieve top-k
   |       cross-encoder verifies equivalence
   |       verified hit? -> return
   |
   +--> Fine-tuned DeBERTa router (prompt only)
           |
           +--> Qwen3-0.6B  (small)
           +--> Qwen3-1.7B  (medium)
           +--> Qwen3-8B    (large)
                   |
                   v
           Hugging Face Inference API  (router.huggingface.co/v1)
                   |
                   v
                Response  --> written to both caches --> dashboard
```

---

## Problem

Serving every prompt with the biggest model is simple and wasteful. Two obvious
savings:

1. **Right-size the model.** "What is 25 * 17?" does not need an 8B model. But you
   only know that *after* you see how the small model does - which you can't do at
   request time. So learn to predict it from the prompt.
2. **Reuse answers.** "What is the capital of France?" and "Which city is the
   capital of France?" deserve the same answer. But a naive embedding-similarity
   cache will also merge "Is aspirin safe **with** warfarin?" with "Is aspirin
   safe **without** warfarin?" - which is dangerous.

## Solution

- **Learned router.** A fine-tuned `deberta-v3-small` 3-class classifier maps a
  prompt to `small` / `medium` / `large`. Trained on labels that mean "cheapest
  good-enough", not "highest score".
- **Verified semantic cache.** Retrieval (TF-IDF + BM25 + dense) proposes
  candidates; a cross-encoder decides whether the cached prompt is *equivalent*
  before its answer is reused. Retrieval alone is never trusted.
- **Exact cache.** A plain Redis hash lookup in front of everything.
- In the **demo app**, the three Qwen models are **only** called remotely through
  the Hugging Face OpenAI-compatible router - nothing but the DeBERTa router runs
  locally. (The training notebook is the opposite: it runs Qwen3 + the 14B judge
  on the GPU to build the dataset, and by default makes no inference-API calls.)

## Architecture

| Layer | What it does | Where |
|---|---|---|
| Exact cache | `SHA256(prompt)` -> stored response | `backend/cache/redis_cache.py` |
| Retriever | TF-IDF + BM25 + dense (numpy), returns top-k candidate prompts | `backend/cache/retriever.py` |
| Verifier | STS cross-encoder + NLI contradiction guard + number guard | `backend/cache/verifier.py` |
| Semantic cache | retrieve -> verify -> hit / miss | `backend/cache/semantic_cache.py` |
| Router | `deberta-v3-small`, softmax confidence + escalation | `backend/router.py` |
| HF client | OpenAI client -> `router.huggingface.co/v1` | `backend/hf_client.py` |
| Cost | estimated $/1k tokens, baseline vs actual | `backend/cost.py` |
| Metrics | in-memory counters for the dashboard | `backend/metrics.py` |
| API | `/chat`, `/metrics`, `/health`, `/adversarial-test` | `backend/main.py` |
| Dashboard | vanilla HTML/CSS/JS | `frontend/` |

---

## Training (`training/meridian_training.ipynb`)

Everything - dataset creation, judging, labelling, dataset push, DeBERTa
training, evaluation, router push - is in **one notebook**. Run it top to bottom
on a single Colab / Kaggle GPU (free T4 is enough).

Run order: the first cell installs vLLM and **removes Colab's torchvision /
torchaudio** (they're built for a different CUDA version than vLLM's torch and
crash on import). Then **Runtime > Restart session** and run from cell `0b`
onward - that cell imports torch + vLLM in a subprocess and prints the versions,
so any install mismatch surfaces immediately instead of 20 min into generation.

```
public datasets
      |
      v
~500 prompts  (GSM8K, HumanEval, MMLU, MBPP, HotpotQA, CNN/DailyMail)
      |
      v
answers from Qwen3-0.6B / 1.7B / 8B     (vLLM, one subprocess per model)
      |
      v
Qwen2.5-14B-Instruct judge -> 0-10     (vLLM on the GPU by default; remote is a toggle)
      |
      v
routing label = cheapest good-enough
      |
      +--> push  srmty/routing_dataset
      |
      v
fine-tune microsoft/deberta-v3-small  (3-class, prompt -> label)
      |
      v
evaluate: accuracy / precision / recall / F1 / confusion matrix / confidence
      |
      +--> push  srmty/meridian
```

### Dataset

~500 prompts, balanced across tasks. Each prompt keeps its `category`.

| source | count | category |
|---|---|---|
| GSM8K | ~85 | math |
| HumanEval | ~85 | code |
| MMLU | ~85 | knowledge |
| MBPP | ~85 | code |
| HotpotQA | ~80 | qa |
| CNN/DailyMail | ~80 | summarization |

### Dataset generation

The notebook writes `gen_worker.py` and runs it **once per model as its own
`python` subprocess** (`small -> medium -> large`). Cycling several vLLM models
through one Colab GPU in a single kernel is unreliable (VRAM isn't fully released,
and vLLM's engine trips over Jupyter's stdout); a fresh process per model
sidesteps both - the OS reclaims all the VRAM on exit. vLLM's continuous batching
does ~500 prompts in a couple of minutes per model. `Qwen/Qwen3-8B-AWQ` (4-bit,
~5.5 GB) keeps it inside 15 GB; `chat_template_kwargs={"enable_thinking": False}`
plus a `<think>...</think>` strip keeps Qwen3's reasoning trace out of the data.
Each worker appends to `checkpoints/gen_<size>.jsonl` every 128 prompts and skips
ids already there, so a disconnect only costs the current chunk.

> The notebook is the only place a Qwen model is downloaded - the demo app always
> routes to them remotely.

### Judge model

`Qwen/Qwen2.5-14B-Instruct` scores all three answers. `JUDGE_LOCAL` (cell 3)
picks how it runs:

- **`True` (default)** - `judge_worker.py` runs `Qwen/Qwen2.5-14B-Instruct-AWQ`
  (4-bit, ~9-10 GB) with vLLM in its own subprocess, greedy decoding. No
  inference-API calls, ~10-15 min for 500 rows on a T4.
- **`False`** - `Qwen/Qwen2.5-14B-Instruct:featherless-ai` through the HF router
  (reuses the worker's prompt template + JSON parser). No local VRAM; spends HF
  inference credits instead.

Either way it sees the prompt and all three answers, weighs **correctness,
relevance, completeness, clarity**, and returns strict JSON:

```json
{"small_score": 7.0, "medium_score": 8.5, "large_score": 9.0}
```

### Routing labels

The target is **not** "which model scored highest". It is "which is the smallest
model that already cleared the bar":

```python
QUALITY_THRESHOLD = 8.0

if   small_score  >= QUALITY_THRESHOLD: label = "small"
elif medium_score >= QUALITY_THRESHOLD: label = "medium"
else:                                   label = "large"
```

So `small=8.3, medium=9.0, large=9.5` labels as **small** - the small model is
already good enough, the extra quality isn't worth the cost.

### Final dataset fields

`id, prompt, category, small_response, medium_response, large_response,
small_score, medium_score, large_score, routing_label, routing_label_id`

Pushed as [`srmty/routing_dataset`](https://huggingface.co/datasets/srmty/routing_dataset)
(stratified 80/20 `train` / `validation` split, seed 42).

### DeBERTa training

`microsoft/deberta-v3-small` as a 3-class sequence classifier
(`0=small, 1=medium, 2=large`). Input is the **prompt only** - at inference the
production router has nothing else. Class-weighted cross-entropy because labels
are usually imbalanced.

### Evaluation

Accuracy, macro precision / recall / F1, confusion matrix, and softmax
confidence. `ROUTER_THRESHOLD = 0.75`:

```
small   0.91
medium  0.07
large   0.02
-> small   (confidence 0.91  >= 0.75  -> accept)
```

If the top probability is below the threshold, the router **escalates** one size
up (`small -> medium -> large`) instead of taking a low-confidence guess.

### Hugging Face outputs

| repo | contents |
|---|---|
| [`srmty/routing_dataset`](https://huggingface.co/datasets/srmty/routing_dataset) | the full labelled dataset |
| [`srmty/meridian`](https://huggingface.co/srmty/meridian) | fine-tuned DeBERTa router (model + tokenizer) |

```python
model.push_to_hub("srmty/meridian")
tokenizer.push_to_hub("srmty/meridian")
```

---

## The demo application

```
meridian_training.ipynb
        |
        +--> srmty/routing_dataset
        |
        +--> srmty/meridian

Application
    |
    +--> loads  srmty/meridian   (the only local model)
    |
    +--> calls  Qwen3 0.6B / 1.7B / 8B  through  https://router.huggingface.co/v1
```

### Redis exact cache

Key `meridian:exact:<sha256(prompt)>`, value
`{"prompt", "response", "model", "tokens"}`. Same prompt string again -> Redis
HIT, no LLM call. Tracks `exact_cache_hits` / `exact_cache_misses`. If Redis is
down the process falls back to an in-memory dict (with a warning) so the demo
still runs.

### Semantic retrieval

Three cheap retrievers run and their top-k are merged:

- **TF-IDF** cosine (`scikit-learn`)
- **BM25** (`rank-bm25`)
- **dense** cosine over `all-MiniLM-L6-v2` embeddings, (brute-force numpy dot product - the cache is small)

Retrieval is deliberately loose - it is allowed to surface near-misses.

### Cross-encoder verification (two stage)

A single duplicate-question model is either too strict (misses "what is 25 * 17"
~ "compute 25 times 17") or too loose (accepts "aspirin **with**" ~ "aspirin
**without**"). So `verifier.py` runs both `SHOULD-HIT` recall and `SHOULD-MISS`
safety as separate stages:

1. **similarity gate** - `cross-encoder/stsb-distilroberta-base`, scored both
   directions, `min` taken (the model is order-sensitive). Handles diverse
   rephrasings.
2. **contradiction guard** - `cross-encoder/nli-deberta-v3-xsmall`. If either
   direction is a *contradiction*, the pair is not equivalent no matter how
   similar the words. Catches `with`/`without`, `enable`/`disable`, `+`/`*`,
   `start`/`end`, entity swaps.
3. **number guard** - regex. Different numbers / years between the prompts
   (`population ... in 1950`, `10 km` vs `100 km`) -> reject.

`score()` returns the similarity, forced to `0.0` if guard 2 or 3 fires;
`is_equivalent()` thresholds it at `SEMANTIC_THRESHOLD` (0.50). On a 12-paraphrase
/ 12-adversarial eval set: **100% recall, 0 false hits** (the old single-model
setup was 70% / 1). Override the models with `VERIFIER_SIM_MODEL` /
`VERIFIER_NLI_MODEL`; if the NLI model can't load, it degrades to similarity-only
with a warning. `GET /adversarial-test` reports live false hits (**0 / 8**).

### HF inference

`backend/hf_client.py` - one `OpenAI` client against
`https://router.huggingface.co/v1`, `generate(prompt, model_name)` where
`model_name` is `small` / `medium` / `large`. `/no_think` plus stripping
`<think>...</think>` keeps Qwen3's reasoning trace out of responses.

### Cost calculation

Demo pricing (clearly labelled **estimated** - open models are not automatically
free to serve). `MODEL_COST_PER_1K_TOKENS = {small: 0.0001, medium: 0.0003,
large: 0.0009}`, override via env. Per request:

```
actual_cost   = 0 if cache hit else price[model] * tokens / 1000
baseline_cost = price[large]   * tokens / 1000
cost_saved    = baseline_cost - actual_cost
saving_%      = cost_saved / baseline_cost * 100
```

No final savings number is hardcoded - the dashboard computes it from traffic.

### Dashboard

`GET /metrics` feeds: total requests, cache hit rate, exact / semantic hits,
misses, false hits, routing distribution (0.6B / 1.7B / 8B), estimated vs
baseline cost, savings, average router confidence, average latency.

---

## Running it

**Python 3.10 - 3.14.** `requirements.txt` uses version floors, not hard pins, so
pip resolves wheels for whatever interpreter you have (torch 2.9+ ships cp313 /
cp314 builds). No GPU needed - the router is tiny and the Qwen models are remote.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then put your HF_TOKEN in it

# optional - exact cache (skip it and an in-memory dict is used instead)
docker run -d -p 6379:6379 redis:7-alpine

cd backend
uvicorn main:app --reload --port 8000
```

Open <http://localhost:8000>.

First `/chat` on a cache miss downloads the router (`srmty/meridian`), the MiniLM
embedder and the cross-encoder - a few hundred MB, once. If `srmty/meridian` is
unreachable the router logs a warning and falls back to a keyword heuristic so
caching / cost / dashboard still work.

**Weak-router handling.** The demo router was trained on only ~500 examples, so
its softmax confidence is modest (~0.5). Two knobs soften that:

- `ROUTER_THRESHOLD` (`.env`, default 0.75, demo 0.45) - min confidence to accept
  a prediction before escalating one size up.
- `ROUTER_BLEND_BELOW` (default 0.55) - when the model's top probability is under
  this (which, with the demo model, is ~always), `route()` **defers to a keyword +
  length heuristic** instead: `prove`/`derive`/`design`/`analyze`/`compare … in
  detail` or >320 chars -> large; `summarize`/`explain`/`translate`/`how do
  I`/`write a`/`debug` or >140 chars -> medium; bare math, `capital of`, `who
  is`, `define`, or <40 chars -> small; **otherwise medium** ("when unsure, spend
  a little more"). `/chat` and `/route` flag `blended_with_heuristic: true`. Set
  `ROUTER_BLEND_BELOW=0` to disable and trust the model alone. On an 18-prompt
  spread this gives roughly 5 small / 9 medium / 4 large.

Both are stopgaps - once the model is retrained on a bigger dataset it will be
confident enough that the blend rarely triggers.

### API

| method | path | |
|---|---|---|
| GET | `/` | dashboard |
| POST | `/chat` | `{"prompt": "..."}` -> answer + routing + cost |
| POST | `/route` | routing decision only, no LLM call (model + probs + reasoning) |
| GET | `/metrics` | dashboard numbers |
| POST | `/reset` | zero the dashboard counters |
| GET | `/health` | liveness + whether the router loaded |
| GET | `/adversarial-test` | run the verifier-safety pairs |

`POST /chat` response:

```json
{
  "response": "...",
  "model": "small",
  "cache_hit": false,
  "cache_type": null,
  "router_confidence": 0.93,
  "estimated_cost": 0.0001,
  "baseline_cost": 0.0008,
  "cost_saved": 0.0007
}
```

On a semantic hit: `cache_hit: true`, `cache_type: "semantic"`,
`router_confidence: null`, `estimated_cost: 0`.

---

## Demo script

**1 - cheap routing.** Ask `What is 25 * 17?`. The router predicts the small
model is enough -> routed to `Qwen3-0.6B`.

**2 - semantic cache.** Ask `What is the capital of France?`, then
`Which city is the capital of France?`. The second retrieves the first, the
verifier accepts it -> **semantic cache HIT**, no Qwen call.

**3 - semantic cache safety.** Ask `Is aspirin safe with warfarin?`, then
`Is aspirin safe without warfarin?`. Retrieval surfaces the first as a candidate;
the verifier **rejects** it (negation) -> a fresh inference call is made. This is
why Meridian uses retrieval **plus** verification, not a similarity threshold.

---

## Environment variables

```
HF_TOKEN=your_huggingface_token
REDIS_URL=redis://localhost:6379
ROUTER_MODEL=srmty/meridian
ROUTER_THRESHOLD=0.75      # min softmax confidence to accept a route
SEMANTIC_THRESHOLD=0.50    # min similarity for a semantic hit (after the guards)
QUALITY_THRESHOLD=8.0      # judge score that counts as "good enough" (training)
```

Never commit the real `.env`.

---

## Limitations

- **Demo-scale dataset.** ~500 prompts, one judge model. Enough to train a
  believable router, not a production one. The judge is itself an LLM and can be
  wrong or biased toward longer answers.
- **Judge = ceiling.** Labels are only as good as `Qwen2.5-14B-Instruct`'s
  scores, and by default it runs in 4-bit. A bigger / full-precision judge (set
  `JUDGE_HF`, or `JUDGE_LOCAL=False` with a 32B+ `JUDGE_REMOTE`) would raise label
  quality at the cost of VRAM or credits.
- **Answers come from 4-bit / sampled generation.** The 8B is an AWQ 4-bit
  checkpoint and all three sample at `temperature=0.7`, so answers (and therefore
  scores and labels) have some run-to-run noise. Point `QWEN_HF["large"]` at the
  full `Qwen/Qwen3-8B` and use greedy decoding for a cleaner but heavier run.
- **Verifier is small + general-purpose.** `stsb-distilroberta` + `nli-deberta-v3-xsmall`
  score 100% / 0 on a hand-built 24-pair set, but that set is small and English-only;
  they were not tuned for medical or legal equivalence. `SEMANTIC_THRESHOLD` still
  trades recall against false hits.
- **In-memory everything.** Metrics and the semantic index live in the process
  and reset on restart. The retriever re-fits TF-IDF/BM25/embeddings on every insert -
  fine for a demo cache of tens of entries, not for millions.
- **Cost numbers are made up.** They are plausible relative sizes, not provider
  invoices. Set the real `COST_*` values if you have them.
- **Provider coupling.** Model ids pin `:featherless-ai`. If a provider stops
  hosting a model, change the suffix in `.env` / `hf_client.py`.
- **No auth, no rate limiting, single process.** Out of scope on purpose.
