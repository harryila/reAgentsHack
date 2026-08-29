# Anthropic provider and credit policy

Last reconciled against local ledgers: 2026-08-29

The operation table below records the original $50 workshop allocation and remains useful as
historical provenance for those runs; it is not the current cross-project accounting rule. The
operator subsequently authorized hosted-model work up to an approximate hard boundary of $100.
The metadata-only, response-ID-deduplicated reconciliation after the full Fable run is
$38.616150 in usage-reported/accounted cost plus $17.482919 in pessimistic liability for every
unknown or ambiguous attempt, for a pre-native-pilot conservative total of $56.099069. The v3
native-numeric pilot then made two distinct, non-retried generation attempts that both terminated
with HTTP 400 before returning usage. Its local terminal records $0 observed generation cost, but
absence of a usage object is not an invoice-level zero. Until billing reconciliation, the policy
therefore carries the complete $1.932800 certified v3 request ceiling as unresolved liability.

The terminal v4 recovery carried a $1.815360 combined certified upper bound for its successful
source-free canary and two scientific calls. Both scientific responses included provider usage;
their observed cost was $0.306090, but the next-phase prior deliberately retained the complete v4
certified bound. That produced the v5 reconciled prior of $59.847229. V5 temporarily reserved a
further $3.000000, so the peak project reservation was $62.847229. Its two terminal scientific
responses both included usage and were locally accounted at $0.297980 against a $1.286260
scientific certificate. The post-terminal locally accounted total is therefore $60.145209 when the
unused v5 reservation is released, leaving $39.854791 below the operator's $100 boundary. Retaining
the entire v5 scientific certificate instead would give the still-conservative $61.133489. These
are local token-price calculations, not provider invoices.

Unused authorization envelopes are not spend. Every future live command still needs its own
pre-call immutable roster and liability proof, and its new worst-case liability must keep the
cross-project conservative total below $100. At-most-once failures remain charged at their frozen
worst-case liability and are never retried; the older `exact_once` contract name does not imply a
successful call after a crash-poisoned intent.

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

These were per-operation ceilings, not targets. Their old `$50` reserve calculation is superseded
by the cross-project reconciliation above. A fixture or unavailable-human-approval branch spends
$0. The current model names and prices are pinned in `providers.py`; re-check them against Anthropic's
[official model overview](https://platform.claude.com/docs/en/about-claude/models/overview) before a
future live run. The Python client is locked to the current 0.120 release line, whose official
[SDK changelog](https://github.com/anthropics/anthropic-sdk-python/blob/main/CHANGELOG.md) includes
Messages API structured output through `output_config.format`.

### Historical workshop GEPA experiment allocation

For the original NeurIPS workshop experiment, the unused remap allocation was suspended. One extraction-only
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
  ceiling for that historical allocation under the configured reflection batch headroom, not a mechanically hard cross-provider
  ledger; the task-provider portion remains hard after reserving the planned reflection allowance.
- Budget preflight charges the full configured output-token allowance; unknown failures are charged
  at that conservative ceiling. An HTTP 400 before a usage-bearing response records zero *observed*
  generation cost and a sanitized error type, status, and request ID, but retains the request's
  certified liability until provider billing independently confirms zero charge.
- A successful source-free canary may be reused only as an immutable, externally replayed
  prerequisite. Reuse does not authorize retrying a terminal scientific request and must not be
  counted as a new provider charge in later-phase accounting.
- Anthropic responses may contain one JSON `text` block plus allowlisted non-text reasoning blocks.
  The adapter persists only the text and non-text block-type names, never hidden reasoning content.
  Zero or multiple text blocks, unknown block types, refusals, malformed JSON, or failed local
  postvalidation close the request terminally without retry.
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
