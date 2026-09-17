# Deploying OpenJev Phase 1 on Runpod Serverless

The upstream repo is a research package with a JSONL CLI, not a server. The
three files added here — `handler.py`, `Dockerfile`, `test_input.json` —
wrap `openjev_phase1` in a Runpod queue worker.

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

`direct`, `serial`, and `shared` all read from the causal model baked into
the image. `reranker` needs the native reranker weights instead — build a
second image with `--build-arg OPENJEV_MODEL=Qwen/Qwen3-Reranker-4B
--build-arg OPENJEV_REVISION=22e683669bc0f0bd69640a1354a6d0aebcfeede5` and
deploy it as its own endpoint.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENJEV_MODEL` | `Qwen/Qwen3.5-4B` | Baked at build time; must match the weights in the image |
| `OPENJEV_REVISION` | `851bf6e8...` | Pinned 40-char commit, required by `load_causal_model` |
| `OPENJEV_MODE` | `direct` | Mode used when a request omits `mode` |
| `OPENJEV_MAX_TOKENS` | `4096` | Prompt ceiling; the repo refuses to truncate |
| `OPENJEV_MAX_ROWS` | `64` | Rows accepted in one job |

`HF_HUB_OFFLINE=1` is set in the image. Weights are baked in, so a
scale-to-zero cold start never downloads them — override it only if you
switch to a model that is not in the image.

## Option A — Runpod builds from GitHub (recommended)

No local Docker work, and Runpod builds natively on x86_64.

1. Push this tree to a GitHub repo you own (a fork of `TheoLeeCJ/openjev`
   with these files added is fine).
2. In the Runpod console, **Serverless → New Endpoint → Import Git
   Repository**, pick the repo, branch `main`, Dockerfile path `Dockerfile`.
3. Endpoint type **Queue**; GPU and worker settings per the table below.
4. Deploy. Watch the **Builds** tab — the image is roughly 18 GB, so the
   first build takes a while.

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
| Container disk | 30 GB | Image is ~18 GB with the weights baked in |
| Scaler | queue delay, 4 s | |
| Timeout | 300000 ms | |

## Local testing

The handler cannot run on a machine without CUDA — `load_causal_model`
raises before the Runpod SDK starts. Test against the deployed endpoint
instead, using `test_input.json` as the request body.
