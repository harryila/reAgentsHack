# Harvester live-to-frozen validation

This is a small invariant and transport check for the Paperclip-free harvester. It is
not a retrieval benchmark, does not estimate recall, and does not establish the
scientific relevance or correctness of the returned work.

## Fixed protocol

The CLI owns the query and bound; neither is a command-line option:

- provider: the cross-domain OpenAlex Works index;
- fixed computer-science probe: `Attention Is All You Need`;
- live result limit and page size: one;
- full-text order: provider-declared direct OA, Europe PMC, then arXiv;
- replay: exact-query `FrozenCorpusSource`, one result per page; and
- summary: closed metadata-only schema with no title, abstract, or content body.

The command archives the exact live search response and any successful public
full-text response under the gitignored `data/cache/harvester-openalex-validation-v1/`
directory. It then serializes the normalized live document into a frozen metadata
corpus, replays the same query without network access, hashes every normalized field,
and verifies every archive receipt and content blob before writing the paper summary.
Existing outputs are never overwritten. If live transport fails, a metadata-only
failed summary and a gitignored failure record preserve that outcome and the partial
cache remains untouched.

```bash
uv run python scripts/validate_harvester.py --max-attempts 1
```

## Frozen result

The run completed on 2026-08-26 and is recorded in
[`validation_summary.json`](../../artifacts/paper/harvester/validation_summary.json).
OpenAlex returned document `W2626778328`. Its live and replay normalized-document
hashes are both
`6098a3c01b6b9eef4fec26dd6bc4bb54ef60efb6ae717eb89ff70362db47e1b3`.
The archived open PDF is 2,215,244 bytes with SHA-256
`bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697`.
All four archive receipts verified. One provider-declared direct-OA location failed;
that warning was retained, and a subsequent declared OA location returned the PDF.

The summary SHA-256 is
`5d127f1979b71acbd8f400d7892609792dc17f65615d0e33a99d7b1c5f6e6381`.
The ignored frozen corpus SHA-256 is
`d862ebf91d941bc4d91efab2dbea7b72ca1b1f9eb388e711fa903bd635e64697`.
The corpus and raw archive are local reproducibility inputs, not redistributable paper
artifacts; the tracked summary contains only IDs, hashes, byte/media/status metadata,
timestamps, counts, and warnings.

## Claim boundary

This successful check supports only these implementation claims:

1. the bounded public response was normalized and archived;
2. an OA full-text payload could be resolved and hash-locked;
3. the normalized metadata could be frozen and searched offline; and
4. the live and replay normalized identities were exactly equal for this one result.

It provides no denominator, relevance judgments, missed-study accounting, or gold
corpus. Consequently it must not be presented as retrieval-recall, systematic-review
coverage, cross-domain performance, or meta-analysis evidence.
