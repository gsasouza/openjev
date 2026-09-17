# Deploying OpenJev Phase 1 on Runpod Serverless

The upstream repo is a research package with a JSONL CLI, not a server. The
files added here — `handler.py`, `worker_cache.py`, `Dockerfile`, and
`test_input.json` — wrap `openjev_phase1` in a Runpod queue worker.

## What the worker does

`load_causal_model` runs once at cold start and the model stays resident for
the worker's life. Each job scores one or more decision rows and returns the
per-option probabilities exactly as `openjev-score` writes them to JSONL.

`core.load_causal_model` refuses to start unless exactly one CUDA device is
visible, so keep `gpu.count` at 1.

## Request shape

```json
{
  "input": {
    "mode": "direct",
    "max_tokens": 4096,
    "rows": [
      {
        "id": "support-1",
        "state": "The deployment completed at 14:02 UTC. ...",
        "question": "Is there evidence that the deployment succeeded?",
        "options": [
          { "id": "yes", "description": "The deployment succeeded." },
          { "id": "no", "description": "The deployment did not succeed." },
          { "id": "insufficient", "description": "The evidence is insufficient to decide." }
        ]
      }
    ]
  }
}
```

`rows` may be replaced by a single `row` object. `mode` defaults to the
image's `OPENJEV_MODE` (`direct`) and accepts `direct`, `serial`, `shared`,
and `reranker`. Rows are validated with `core.validate_row` before the model
is touched, so a malformed row comes back as `{"error": "ValueError: ..."}`
rather than burning GPU time.

`direct`, `serial`, and `shared` all read from the one causal model the
worker loads. `reranker` needs the native reranker weights instead, so it
belongs on its own endpoint: build with `--build-arg
OPENJEV_MODEL=Qwen/Qwen3-Reranker-4B --build-arg
OPENJEV_REVISION=22e683669bc0f0bd69640a1354a6d0aebcfeede5` and set that
endpoint's Model field to `Qwen/Qwen3-Reranker-4B`.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENJEV_MODEL` | `Qwen/Qwen3.5-4B` | Must match the endpoint's **Model** field, so the cache lookup resolves |
| `OPENJEV_CACHE_ROOT` | `/runpod-volume/huggingface-cache/hub` | Where Runpod mounts cached models |
| `OPENJEV_REVISION` | `851bf6e8...` | Pinned 40-char commit, required by `load_causal_model` |
| `OPENJEV_MODE` | `direct` | Mode used when a request omits `mode` |
| `OPENJEV_MAX_TOKENS` | `4096` | Prompt ceiling; the repo refuses to truncate |
| `OPENJEV_MAX_ROWS` | `64` | Rows accepted in one job |

## Where the weights come from

The image does **not** carry the weights. Runpod's model cache supplies them,
and the endpoint's **Model** field must be set to `Qwen/Qwen3.5-4B` for that to
happen — without it, workers fail at startup with a message naming the missing
cache path.

This was not the first design. Baking the weights in produced an ~18 GB image
that took over 25 minutes to reach a worker and never finished starting: the
build was fine, distribution was not. Runpod also lists cached models as the
preferred approach over baking.

`worker_cache.resolve_cached_snapshot` looks for the cached copy at:

```
/runpod-volume/huggingface-cache/hub/models--Qwen--Qwen3.5-4B/snapshots/<commit>/
```

It asks for the **pinned** commit by name rather than following `refs/main`, so
a cache that has moved on to a newer revision is rejected loudly instead of
silently scoring against different weights. That directory is then handed to
`core.load_causal_model` as a local path, which satisfies the repo's rule that
remote models carry a 40-character revision.

`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set in the image so a cache
miss fails fast instead of pulling 8 GB on every cold start.

## Option A — Runpod builds from GitHub (recommended)

No local Docker work, and Runpod builds natively on x86_64.

1. Push this tree to a GitHub repo you own (a fork of `TheoLeeCJ/openjev`
   with these files added is fine).
2. In the Runpod console, **Serverless → New Endpoint → Import Git
   Repository**, pick the repo, branch `main`, Dockerfile path `Dockerfile`.
3. Endpoint type **Queue**; GPU and worker settings per the table below.
4. Set the **Model** field to `Qwen/Qwen3.5-4B` so Runpod caches the weights.
   The worker fails at startup without it.
5. Deploy. Watch the **Builds** tab.

## Option B — build locally and push to a registry

Only worth it if you want the image under your own account. On Apple
Silicon this is a cross-build and will be slow.

```bash
docker buildx build --platform linux/amd64 \
  -t docker.io/<user>/openjev-worker:v1 --push .
```

Then the endpoint can be created straight from the image reference.

## Endpoint settings

| Setting | Value | Why |
| --- | --- | --- |
| GPU pool | `AMPERE_24` (RTX 3090 / A5000 / L4, 24 GB) | 4B at bf16 is ~8 GB of weights; the README's own target is a 3090 |
| GPU count | 1 | `load_causal_model` requires exactly one visible CUDA device |
| Min CUDA | 12.8 | Matches the `cu128` torch wheel in the image |
| Workers | min 0, max 3 | Scale to zero; pay only while a job runs |
| Container disk | 30 GB | Headroom for the image plus the cached weights |
| Scaler | queue delay, 4 s | |
| Timeout | 600000 ms | |

## Local testing

The handler cannot run on a machine without CUDA — `load_causal_model`
raises before the Runpod SDK starts. Test against the deployed endpoint
instead, using `test_input.json` as the request body.
