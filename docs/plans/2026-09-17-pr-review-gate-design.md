# PR review gate on OpenJev

Design, 2026-09-17. Status: validated, not implemented.

## Problem

Sending every pull request to a frontier model for review is expensive and
slow, and most hunks in most PRs violate nothing. A cheap, fast, deterministic
filter in front of that model would cut the bill to what actually needs
judgement.

## Why OpenJev fits

OpenJev does not generate text. It reads option probabilities from a single
forward pass, so it cannot author a review. It can answer a closed question —
"does this hunk violate rule X?" — in about 70 ms, deterministically.

That makes it a router, not a reviewer. It decides what deserves a reviewer's
money. The expensive model still writes every finding a human reads.

## Architecture

```
PR diff ──▶ [1] Chunker ──▶ [2] OpenJev gate ──▶ [3] LLM reviewer ──▶ report
              CPU, free       GPU, ~70ms/rule     expensive, rare
```

### 1. Chunker

Parses the diff into hunks. For each, builds a structured `state`:

```json
{"file": "src/auth.ts", "lines": "42-58",
 "added": ["  const t = req.headers.token"],
 "removed": ["  const t = getToken(req)"]}
```

Structured JSON rather than a raw patch drops `@@` headers and repeated path
noise. Send changed lines plus roughly three lines of context.

It then selects the rules whose glob matches the file, and emits one job per
hunk containing every applicable rule as a row — all rows sharing that exact
`state`.

### 2. Gate

Each job is posted with `mode: "shared"`. `score_shared` encodes the state
prefix once, prefills the KV cache once, replicates it across branches with
`reorder_cache`, and scores every rule suffix in one batched forward.

The hunk is therefore paid once per hunk, not once per hunk-times-rule. For an
800-token hunk against 25 rules:

| Mode | Tokens through the model |
| --- | --- |
| `direct`, 25 full prompts | 25 x ~900 = ~22,500 |
| `shared`, 1 prefill + 25 suffixes | 800 + 25 x ~50 = ~2,050 |

About 11x. That difference is what makes the gate affordable rather than a
luxury, and the whole design rests on it.

### 3. Escalation

A hunk-rule pair is cleared only when `P(violates)` falls below
`CLEAR_THRESHOLD`. Everything else escalates: the hunk, the rules that did not
clear, and their scores go to the LLM, which writes the actual prose.

The gate never authors findings and never blocks a build. It is deliberately
biased toward escalation.

## Rules

```yaml
- id: auth-no-raw-header-token
  aspect: security
  glob: ["src/**/*.ts"]
  question: "Does the change read an auth token directly from headers instead of the token helper?"

- id: err-no-silent-catch
  aspect: error-handling
  glob: ["**/*.ts", "**/*.py"]
  question: "Does the change swallow an error without logging or rethrowing it?"
```

Seed the file from `CLAUDE.md`, `CONTRIBUTING.md`, and existing style guides,
then hand-edit. Prose has to be compiled into closed questions somewhere; a
checked-in file makes that step explicit and reviewable.

Options are fixed across every rule:

```json
[{"id": "violates",  "description": "The change violates this rule."},
 {"id": "complies",  "description": "The change does not violate this rule."},
 {"id": "unclear",   "description": "The hunk alone is insufficient to decide."}]
```

`unclear` is load-bearing. `serial` mode reports `allowed_token_mass`, which
says how much probability mass landed on the declared options rather than the
rest of the vocabulary — a natural abstention signal. `shared` mode does not
report it. The explicit `unclear` option recovers that abstention: the model is
never forced to take a side on a hunk that cannot support one. Treat `unclear`
as escalate, never as clean.

Reporting axis: `correctness`, `error-handling`, `security`, `naming`,
`testing`, `complexity`, `api-compat`, `performance`.

### Constraints the rules must respect

- At most 16 options, and each answer letter must be one exact round-trip
  token, or `_slot_ids` raises. Three options is well inside that.
- Keep `question` strings similar in length. `_suffix_layout` pads every suffix
  to the longest one, so a single rambling rule taxes the whole batch. Watch
  `padded_suffix_tokens` against `true_suffix_tokens` in `shared_timing` and
  rewrite the outliers.
- Rules must be answerable from one hunk. "Is this function tested?" cannot be
  answered from a diff and will produce noise. "Does this change add an
  exported function without a corresponding test file change?" can be, if the
  PR's changed-file list is included in `state`.

## Batching and limits

One job per hunk. The handler caps `OPENJEV_MAX_ROWS` at 64, and that cap
matters: splitting one hunk's rules across two jobs re-prefills the hunk and
forfeits the saving the design depends on. Keep applicable rules per file at or
below 64, or raise the cap. Hunks are independent, so run them concurrently up
to `workers.max`.

## Failure handling

| Condition | Source | Behaviour |
| --- | --- | --- |
| Hunk plus rule exceeds 4096 tokens | `encode_prompt` raises; it refuses to truncate | Split the hunk; if still too large, escalate unconditionally |
| Mixed states in one job | `score_shared` raises | Grouping bug: fail loudly in tests |
| Malformed row | `validate_row` | Handler returns `{"error": ...}`; escalate |
| Endpoint down or cold | scale-to-zero | Escalate everything; degrade to plain LLM review |
| `unclear` wins | model | Escalate |

Every path escalates. A recall-first gate must never convert an error into a
clean bill of health; that single bug would defeat the design silently.

## Cost

A 40-hunk PR against 25 rules is 40 prefills of roughly 800 tokens plus 1,000
suffixes of roughly 50. Under a minute of GPU at $0.69/hr — about a cent per
PR. The LLM then sees only what did not clear.

## Testing

The chunker, rule matching, job grouping, and threshold logic are pure
functions and belong in unit tests that never touch a GPU. Add one
recorded-fixture test: a known diff, stored scores, and the asserted escalation
set, so threshold regressions are caught without spending GPU in CI.

## Open questions

- `CLEAR_THRESHOLD` has no principled value yet. `probability_status` states
  the scores are "conditional option score; uncalibrated as decision
  confidence", so the threshold must be fitted against labelled PRs rather than
  guessed. Until then, set it low and accept a high escalation rate.
- Escalation rate is unmeasured. If it exceeds roughly half of hunks the gate
  is not paying for itself, and either the rules or the threshold need work.
- Rules are unproven. A rule that fires on everything is worse than no rule.
