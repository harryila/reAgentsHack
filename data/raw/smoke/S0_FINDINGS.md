# s0 Smoke Probe Findings — 2026-08-15 (archived run; exact run timestamp not recorded)

## GATE G1: PASSED
- `map --output-schema` with nested findings ARRAY works on the DEFAULT (quick-reader) worker.
- Probe: m_2bc51e4b over 4 papers from s_8b448463. RCT (PMC12384908) → eligible=true,
  5 findings (one per outcome), each with dose_raw, population, sample_size, timepoint_raw,
  verbatim evidence_quote, evidence_lines (["L18"] format). Directions correct incl. no_effect.
- Exact raw boundary: 4 model responses and 6 raw findings total — the 5 RCT findings plus 1
  schema-valid review-pollution finding copied from a References title. Normalization yields 6
  accepted FindingRows, 0 quarantined rows, and 1 `section_flagged=true` row; therefore only the 5
  RCT rows are non-section-flagged candidates for later validation—not a production headline
  cohort. The other 2 papers are clean ineligible zero-finding responses.
- Zero-finding behavior: genuinely ineligible papers return
  `{eligible:false, exclusion_reason, findings:[]}`; the polluted review is the documented
  counterexample that makes deterministic screening and section flags mandatory.
- Latency: 4 papers in 8.0 s wall (1.3–4.6 s/paper, parallel). 300 papers ≈ minutes at this rate.

## Gated features
- `--worker structured-extraction|eligibility-screen|exhaustive-extraction` → "limited to GXL
  testers". ASK ORGANIZERS for tester enablement. Default worker suffices meanwhile.

## Retrieval semantics (changes triage design)
- Default search = recency-weighted slice (all hits 2024–2026). Use `--all` for full corpus.
- `-c` returns min(n, hits), NOT a true count. True counting: `paperclip sql` —
  TRE title-match: 229 PMC (73 pre-2023); antioxidant/vitC/vitE × exercise/training: 315 PMC.
- Corpus pub_date spans 1781–2027.
- Map: `--resume MAP_ID`, `--retry-failed`, `--cancel`, `-j` (default 32, cap 256). Guidance:
  one big map with -j, not offset windows. Overnight runner = thin wrapper around --resume.

## Anchor coverage — GAPS CONFIRMED
- ABSENT from corpus (by doi, pmid, exact title, SQL ILIKE): Paulsen 2014 (J Physiol),
  Lowe/TREAT 2020 (JAMA IM), Liu 2022 (NEJM). Paywalled-journal PMC deposits are missing.
- Present: e.g. PMC10611992 (TRE weight loss RCT 2023), ChronoFast secondary analyses, etc.
- Consequence: pick audit anchors FROM the corpus at triage; state coverage honestly in demo.

## Live failure-mode confirmation
- Review PMC12845069 slipped eligibility (model marked eligible=true and extracted a finding
  from a CITED study, evidence lines pointing at a reference title). Confirms need for:
  deterministic article_type screen BEFORE map + evidence-section flag + prompt hardening.

## Scientific bonus
- PMC12384908: vit C+E ENHANCED resistance-training gains in older sarcopenic women (2025 RCT) —
  opposite direction to the young-athlete blunting literature. Age/population moderator signal
  visible in a 4-paper probe.

## Keys
- PAPERCLIP_API_KEY + ANTHROPIC_API_KEY in .env (gitignored), both verified working.
