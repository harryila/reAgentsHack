# Anthropic provider and credit policy

Last verified: 2026-08-26

The ordinary pipeline is offline. Unit tests, fixture generation, fixture analysis, export replay,
and the Streamlit app use deterministic local data and must never instantiate an Anthropic client.
Every real call requires an explicit `--live` flag and an exact archived request key.

## Model roles

| Operation | Model | Effort | Purpose | Operation ceiling |
|---|---|---:|---|---:|
| wiring smoke | `claude-sonnet-5` | low | one tiny synthetic transport/schema check | $0.25 |
| corpus baseline | `claude-sonnet-5` | low | one visibly ungrounded paragraph per cohort | $1.00 |
| quote-to-direction verification | `claude-opus-5` | high | independent scientific entailment check | $15.00 |
| approved remap proposal | `claude-fable-5` | high | one capability-sensitive proposal after human approval is available | $10.00 |
| approved full-cohort remap | `claude-opus-5` | high | structured echo-back extraction only | $15.00 |

These are ceilings, not targets. The remaining $8.75 of the $50 credit allocation is reserve. A
fixture or unavailable-human-approval branch spends $0. The current model names and prices are
pinned in `providers.py`; re-check them against Anthropic's
[official model overview](https://platform.claude.com/docs/en/about-claude/models/overview) before a
future live run. The Python client is locked to the current 0.120 release line, whose official
[SDK changelog](https://github.com/anthropics/anthropic-sdk-python/blob/main/CHANGELOG.md) includes
Messages API structured output through `output_config.format`.

### Workshop GEPA experiment allocation

For the NeurIPS workshop experiment, the unused remap allocation is suspended. One extraction-only
GEPA run may use at most $8 for task rollouts and $3 for reflection. Task rollouts remain inside the
archived `AnthropicProvider` ledger. GEPA 0.1.4 reflection uses LiteLLM with zero retries and a fixed
output-token cap. Each active prompt kind receives a fresh tracked LM and a $2.50 cost stopper; the
CLI's explicit per-kind batch-headroom reservation covers the final call that may cross a checked
threshold. Before loading credentials, the CLI requires existing archived provider estimates plus
the complete task ceiling plus every per-kind reflection stopper and headroom reservation to remain
below the $50 planning ceiling. It reduces the task provider's global ledger limit by the reflection
reservation. Claude Sonnet 5 must not receive a non-default temperature; Claude Sonnet 4.6 may be
used as the reflection model when LiteLLM's bundled price map does not yet recognize Sonnet 5.

## Enforced boundaries

- `FixtureProvider` is the only provider used by default tests.
- `AnthropicProvider` refuses calls unless `live_enabled=True`.
- Every request is archived before its result can be reused, and the same operation/request key is
  never attempted twice—even after malformed JSON or a non-terminal response.
- Each operation has a local cost ceiling. Ordinary operations use a recursive ledger rooted at
  `data/raw/providers/`; the GEPA CLI scans the repository root so its gitignored benchmark-text
  receipts and ordinary receipts share one task-provider ledger. Reflection spend is tracked by
  GEPA's LiteLLM wrapper, not that receipt scan. Therefore `$50` is a conservative combined planning
  ceiling under the configured reflection batch headroom, not a mechanically hard cross-provider
  ledger; the task-provider portion remains hard after reserving the planned reflection allowance.
- Budget preflight charges the full configured output-token allowance; unknown failures are charged
  at that conservative ceiling. A confirmed HTTP 400 rejection before generation records zero
  usage and zero cost, while retaining a sanitized error type, status, message, and request ID.
- Structured operations retain their full local JSON Schema as the evaluator contract. Immediately
  before `output_config.format`, the provider transforms a defensive copy with the official SDK's
  [`anthropic.transform_schema`](https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/lib/_parse/_transform.py).
  The archive records both schemas, both hashes, and the SDK version. Parsed output is then
  validated locally against the original schema, so moving unsupported wire constraints into
  descriptions does not weaken evaluation. Missing, duplicate, or unknown IDs still fail; they
  never become agreement.
- Provider archives contain prompt/model/usage/output metadata but never API keys. `.env` is ignored
  and must remain mode `0600`. Archived API exception details are allowlisted and credential-like
  strings are redacted.
- The frozen app imports no provider, Paperclip, or HTTP client and reads only its validated demo
  directory.

## Live-call checklist

1. Finish the entire corresponding fixture path and run the offline test/lint gates.
2. Inspect the exact prompt, schema, request count, model, effort, max tokens, and conservative cost.
3. Confirm no archive already exists for the request key.
4. Use the smallest role-appropriate model above; do not use a high-capability model for transport
   wiring.
5. Run one call. Inspect the archived usage and response before authorizing the next distinct
   operation.
6. If a call fails, keep its archive and surface the typed unavailable/failed state. Do not retry the
   same cohort to obtain a more favorable answer.

The one-time transport check is:

```bash
uv run python scripts/anthropic_smoke.py --live
```
