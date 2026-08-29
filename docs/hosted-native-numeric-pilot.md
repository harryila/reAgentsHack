# Hosted native-numeric yield pilot

This diagnostic asks a deliberately narrow engineering question: can the hosted
native extractor return an officially postvalidated, target-associated, exactly
source-grounded binary `NativePublicationExtraction` for each of two purposively
selected public PMC full-text records? Selection used source text only. The endpoint
is yield, not accuracy, representativeness, synthesis quality, scientific truth, or
claim-release authority.

## Terminal v3 result

The immutable v3 workspace attempted each of its two frozen requests once. Both calls
returned HTTP 400 before any usage-bearing response and were terminally closed with no
application or SDK retry:

- counted inputs: 15,344 and 12,048 tokens;
- certified combined request ceiling: $1.932800;
- completed native extractions: 0/2;
- failed or ambiguous extractions: 2/2, both `provider_http_400`;
- observed generation cost in the terminal: $0;
- accuracy, representativeness, synthesis, and release authority: false.

The terminal run SHA-256 is
`00c3895c7ed1b8803253a93965708748ed48bc6cc8eb225574ad80f44d8fa511`;
the terminal-report SHA-256 is
`e28dfe208e54823af895be361d622c8ec602c5afa1199f297caae92f8599a04e`.
The offline bridge correctly converted the complete zero-yield run into a standard v4
grounding package with zero completed fragments. Its bridge-receipt SHA-256 is
`343b0fbd7cf935e2e37becb0d0c290d69f78b4c014c66448854b535ee3021ba7`
and its package SHA-256 is
`13981926a266acc5128299bf6ef0e9fc41622cb609cdf8b786c2c26cbee6aff9`.

The $0 value is an observed local value, not an invoice claim. Because the failed
responses contained no usage object, project accounting retains the entire $1.932800
certified ceiling as unresolved liability until provider billing confirms otherwise.
The exact request identities are exhausted and must never be retried.

## Failure diagnosis

This result is not evidence that the model failed to read either paper. Both requests
were rejected before a scientific response existed. The shared request envelope uses
an installed SDK-supported model, effort, service tier, and Messages API shape already
used by successful Fable calls elsewhere in the evaluation. The differing document
text therefore does not explain the common HTTP 400.

The dominant diagnosis is provider grammar compilation. The generation schema was
7.84 KB with 394 nodes, maximum nesting depth 17, 80 required properties, and 12
arrays. The SDK-transformed form also removed every `maxItems` constraint, leaving 11
nested arrays with `minItems` but no maximum. Anthropic's
[structured-output documentation](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
describes separate grammar compilation and notes that token counting does not compile
the grammar. Both generation calls failed when that grammar first became necessary.
The runtime intentionally retained only allowlisted incident metadata, so this is a
high-confidence inference rather than a quotation of the server error body.

## Fresh v4 recovery contract

V4 is a new execution identity, not a retry. Before any source-bearing call it must
pass one source-free, at-most-once canary using the same model and prompt-JSON
transport. The two scientific calls are eligible only after the canary terminal is
externally replayed and hash-bound into their plan.

The scientific model still has to emit the complete native extraction object. V4
places the original closed candidate schema in a canonical terminal system-message
envelope and omits `output_config.format`, so the provider does not compile the large
grammar. The original schema retains all fixed-value and exact-list-size constraints.
Every returned object then has to pass, locally and fail closed:

1. strict JSON parsing with duplicate-key and non-finite-value rejection;
2. the complete candidate schema;
3. the unchanged official native extraction schema;
4. canonical Pydantic round-trip equality;
5. exact arm/count association to the frozen target clause; and
6. exact source grounding through the standard native grounding verifier.

V4 uses Fable 5 at high effort, zero retries, and a 10,240-token output ceiling. The
source-free canary and both scientific requests share a combined $3.00 worst-case
phase ceiling. Together with the unresolved v3 ceiling, the pre-v4 conservative
project total is $58.031869 against the operator's $100 hard boundary.

Even a successful 2/2 v4 result would establish only executable, source-grounded
native extraction on two purposively selected records. It would not estimate
extraction accuracy or authorize a scientific claim.

## Terminal v4 result

The source-free prompt-JSON canary passed on its first and only identity. Its semantic
terminal SHA-256 is
`4ca9fefe979c4194c4eda7e079428a063881d0264f202ac4841f22018dd2c7f6`,
and its charged/certified upper bound was $0.532210. The two scientific requests then
closed terminally with zero retries:

- counted inputs: 13,576 and 10,291 tokens;
- certified scientific ceiling: $1.283150;
- completed native extractions: 0/2;
- observed scientific cost: $0.306090;
- PMC2427034: provider refusal with no content blocks;
- PMC3104134: one reasoning block plus one JSON text block, rejected by the v4 adapter
  because it incorrectly required the response to contain only one block in total.

The v4 run SHA-256 is
`1510d6ee1489e6d4f587b6a57b91fb7bb96bb0e386fa0824214a02a6e904044a`;
its terminal SHA-256 is
`161a53360a96e734a1cf83c32b2536a0e18923e09a2770d8da805756c957df92`.
The zero-yield bridge receipt is
`dfd8b3a926491bf2a70cdc5a5757e54f53483283acf67a3ccaf9861246a41ae8`,
and the grounding package is
`48b60638f720b338b18b15164fd683d4886168749decd47336670d88286c5112`.
These identities are exhausted and must never be retried.

## Terminal v5 recovery result

V5 is a new prompt, request-key, schema-ID, runtime, and pipeline identity. It reuses
the already successful source-free v4 canary as a hash-bound prerequisite; it is not a
retry of either v4 scientific request. The v5 adapter accepts exactly one JSON text
block while allowing only declared non-text reasoning block types, and the prompt adds
a deterministic source-only target clause before the complete frozen full text.

The immutable v5 run produced one complete native extraction:

- counted inputs: 13,698 and 10,480 tokens;
- certified scientific ceiling: $1.286260;
- observed and locally accounted scientific cost: $0.297980;
- completed native extractions: 1/2;
- PMC2427034: terminal provider refusal, with zero output tokens;
- PMC3104134: one `thinking` block and one JSON `text` block, followed by successful
  candidate-schema validation, official-schema validation, exact target association,
  and source-grounding replay;
- accepted counts: 20/152 in the eradication arm and 123/155 in the placebo arm;
- accepted extraction SHA-256:
  `73fa010ea178738bd2f4110b26a5831a5f788b00d83eef7c710c3012f8247296`.

The v5 plan SHA-256 is
`57874617e629fea6d937cc9f45c353969eab8e584f9af2fa27001dfcda003cdb`;
the hosted run SHA-256 is
`7b775c9923b134899e14bb79796f996ae7da07ebdcfa6324b1c653b728669fc8`;
and the terminal SHA-256 is
`05205c5a508b1e7c34bf0df95622e7bd2a3138a80b58f28e41484e23eec60ece`.
The standard bridge retained both terminal publication states, emitted one estimable
fragment, and externally replayed package v4. Its bridge receipt is
`f31147932fb9026ef5c4f724bb3a629e30aeb5235590737986c5dcebfc782ee8`,
and its package SHA-256 is
`1098b1c92bf9e0033d48a077ef51d450d40f4f95891f27e0892103d7c3e428ac`.

The public `lm verify` path now accepts `hosted_exact_once` as a valid replayed native
execution mode and produced certificate
`171ab569c16e2aaae1a739d98d2e4cabdf081d1ffc6a647bc09b3d1f9007272e`.
It correctly abstained: one estimate cannot satisfy the two-cohort synthesis minimum,
one source is non-estimable, cohort reconciliation is incomplete, the extraction-only
pipeline identity differs from the release pipeline, and no complete-question adaptive
calibration or human audit exists. This is positive end-to-end integration evidence and
one real grounded extraction, not extraction-accuracy, synthesis-effectiveness, or
claim-release evidence.
