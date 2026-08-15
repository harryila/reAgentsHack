VERDICT: pass-with-fixes

[must-fix] WHERE: §2 architecture (s6 line + B4 versioning guard) vs §4.1 (finding_id/remap) vs §4.4/D7 (gain test)
ISSUE: The remap merge path is undefined and the obvious implementation contradicts the version-mixing guard. s6's line says 'merge via finding_id enum' but names no output artifact and no owning stage for producing the merged analysis dataset the 'cross-validated gain test' needs. findings.parquet is 'regenerated from jsonl, never edited' (s4), and s4 hard-fails on more than one distinct (prompt_version, schema_version, cfghash) tuple — a remap run necessarily has a new cfghash, so folding remap output into findings.jsonl and re-running s4 trips the guard by design. Implementer A will append rows to jsonl and use --allow-mixed; implementer B will build a side table joined at s5. These are incompatible, and this sits on the demo-critical M4.5 one-hour window (10:30–11:30 PM).
FIX: Pin it: s6 writes a validated side table (e.g. data/processed/<qid>/remap_<field>.parquet, columns: finding_id, value, plus its own version tuple in a sidecar), s5 gains a --with-remap flag that left-joins on finding_id (base rows keep their original tuple, so no guard conflict). State whether remap output passes through the same pre-pydantic normalizer/quarantine path.

[must-fix] WHERE: §7 milestone table (no s7 row) vs §2 ('app reads artifacts/<qid>/demo/ only') vs M3.5, M4.5, M5, M6
ISSUE: s7 export_demo is never scheduled, yet three deliverables need its output before it plausibly runs: M3.5 (≤8 PM) requires the dashboard to render 'fixture artifacts', but the dashboard reads only artifacts/<qid>/demo/ + manifest schema_version — which only s7 produces; M4.5's 'backup demo video recorded' (10:30–11:30 PM) is worthless unless s7 has run on REAL v1 data by then; M5 claims 'demo fully shippable from 11 PM freeze'. But M6 (Sun ≤9 AM) 'real-data swap verified' implies the real s7 run is Sunday. As written, the Saturday backup video shows fixture data, or the implementer must guess when the real s7 run happens.
FIX: Add explicit s7 runs to the table: (a) s7 on fixtures between M2.5 and M3.5; (b) s7 on real v1 immediately after M4/M4.5 and before the backup video and 11 PM freeze; make M6 a re-run + verification of the swap, not the first real export.

[must-fix] WHERE: §7 M4 row ('headline moderator passes permutation', ≤10:30 PM, 'fail → variant B') vs §7 pattern-failure branch ('no moderator passes permutation p < 0.1 by 11 PM') vs §6 (per-moderator p in table, Westfall–Young adjusted p in demo)
ISSUE: The gate that decides whether Sunday's product is the main narrative or variant B is ambiguous on two axes: (1) is the p < 0.1 criterion the raw per-moderator permutation p or the Westfall–Young family-wise adjusted p? §6 maintains both and they will disagree exactly in the marginal cases where this gate matters; (2) the M4 row fires variant B at ≤10:30 PM while the branch text says 'by 11 PM'. Two implementers will code incompatible gate checks and could ship different demos.
FIX: One sentence in §7: 'Gate = Westfall–Young family-wise adjusted p < 0.1 (or explicitly: raw per-moderator p), computed at M4 (≤10:30 PM); final go/no-go call on variant B at 11 PM.'

[should-fix] WHERE: §2 B4 paragraph (cfghash in run.json sidecar + raw filenames) vs §4.1 ('cfghash (pipeline-filled)') vs §2 fixture spec (~60 rows) vs M2.5
ISSUE: No stage is named as the one that stamps (prompt_version, schema_version, cfghash) onto each findings.jsonl row, yet s4's guard checks those fields on its input rows — so it must be s3's normalizer, but that's only derivable, not stated. Worse, the fixture row spec never mentions the version tuple, so at M2.5 the fixture either fails s4's guard or the guard silently can't run, costing debugging time at 6:30 PM.
FIX: State: s3's normalizer copies the tuple from the run.json sidecar onto every emitted row; the synthetic fixture carries one consistent dummy tuple (e.g. prompt_version='fixture-1') on all 60 rows.

[should-fix] WHERE: §2 (dashboard reads demo/ only) vs §4.1 (quarantine_rate > 10% banner), D5 (exclusion rate next to every entropy figure), §5.3/§9 (audit counts + Wilson, cross-model agreement rate, section-flag rate), §4.4 (s7 inventory = 'frozen artifacts + manifest.json + baseline.json + funnel counts')
ISSUE: The dashboard is barred from reading anything outside artifacts/<qid>/demo/, but several numbers it must display are never assigned to any demo/ file: quarantine_rate, mixed/unclear exclusion rates, human-audit raw counts, cross-model agreement rate, section-flag rate — and the evidence cards need the row-level findings table itself inside demo/. 'Frozen artifacts' is not an inventory. Implementer A will have Streamlit read data/extracted/quarantine.jsonl directly (violating the rule), implementer B will extend manifest.json.
FIX: Enumerate s7's exact output file list (e.g. findings.parquet copy, analysis/* copies, audit.json, verification.json) and declare that all scalar rates live in manifest.json counts.

[should-fix] WHERE: §2 s2 (local fuzzy dedupe via rapidfuzz) vs D3/s3 (extraction = --limit/--offset windows over one Paperclip result ID; checkpointed runner keyed by paper_id)
ISSUE: s2's fuzzy-dedupe and screening decisions are local, but they cannot be expressed in the server-side result ID that s3 windows over. No stage is defined to keep screened-out/deduped papers out of extraction: skip-list in the runner vs extract-everything-and-drop-at-s4 are both defensible readings, with different overnight API budgets and different funnel-count semantics (searched → deduped → screened → findings).
FIX: State: s2 emits include_paper_ids.json / exclude_paper_ids.json; the s3 checkpointed runner (already keyed by paper_id) uses the include list as its work queue and never maps excluded ids.

[should-fix] WHERE: §5.2 Gate G2 floors ('probe entropy interval reaching ≥0.4' and 'est. ≥40 usable primaries') vs G3 phrasing ('interval overlaps ≥0.4')
ISSUE: Two underspecified thresholds decide the 5 PM topic lock: (1) does 'interval reaching ≥0.4' mean upper bound ≥0.4 (lenient, matches G3's 'overlaps') or lower bound ≥0.4 (strict)? Different readings flip the gate on a 10-paper probe; (2) 'est. usable primaries' has no estimator — count-gate total × probe usable-rate? raw merged count? Two implementers get different G2 outcomes.
FIX: Pin both: 'upper bound of the paper-bootstrap entropy interval ≥ 0.4' (same wording in G2 and G3), and 'est. usable primaries = deduped screened count × probe usable-fraction'.

[should-fix] WHERE: D10 tie-break ('corpus grep shows the dose moderator clears projected min-support on both sides of the ~500 mg threshold')
ISSUE: 'Projected min-support' is used in a topic-lock decision but never given a number. §6 offers three different candidate thresholds (<k papers per subset, class in <2 papers, <3 papers per leaf), all defined for post-hoc analysis, not for a grep count. Two implementers pick different N and lock different topics.
FIX: Pin one number in D10, e.g. 'grep-estimated ≥5 distinct papers on each side of 500 mg/day'.

[should-fix] WHERE: §2 s6/s7 lines vs §4.4 (trace.json and baseline.json)
ISSUE: Two demo artifacts have conflicting producers/locations. trace.json: listed on s6's output line (under data/raw/map/<qid>/?) AND as a §4.4 demo artifact in artifacts/<qid>/demo/. baseline.json: architecture lists it as an s7 export output, but §4.4/M4.5 say it is 'generated once tonight' via a one-shot LLM consensus call at 10:30–11:30 PM — before s7's real run — and s7 is otherwise a pure freeze/copy stage; if s7 regenerates it Sunday it violates the 'generated once and archived' requirement.
FIX: State: s6 writes trace.json to artifacts/<qid>/analysis/ (or its own dir); a small dedicated script (or an s7 --gen-baseline one-time flag) makes the baseline LLM call Saturday night; s7 only copies both into demo/ and refuses to regenerate baseline.json if present.

[nit] WHERE: Design header + M7 ('freeze 9:45 AM, submit 10:15 AM PT') vs research-notes.md §1 ('Submission deadline: Sunday Aug 16, 10:45 AM PT')
ISSUE: The design's schedule ends 30 minutes before the verified deadline with no note saying the buffer is intentional; a fresh implementer burns time reconciling which is authoritative.
FIX: Add '(30-min deliberate buffer vs the 10:45 official deadline)' to the header or M7 row.

[nit] WHERE: §1 cut list / §7 cut order ('funnel-plot stretch') vs §4.4/§10 ('funnel counts', 'funnel 4-bar')
ISSUE: Two unrelated things share the name 'funnel': the core corpus-funnel 4-bar (manifest counts, must keep) and the publication-bias funnel plot (15-min stretch, first-ish to cut). Under time pressure the cut-order line could be misread as cutting the 4-bar.
FIX: Rename in the doc: 'corpus 4-bar' vs 'bias funnel plot (stretch)'.

[nit] WHERE: §2 s1 ('merged, deduped paper list') vs s2/D4 (DOI/PMID + fuzzy dedupe ownership)
ISSUE: s1 already claims 'deduped' while s2 owns dedupe; the kind of dedupe s1 does is unstated, blurring stage ownership.
FIX: One clause: 's1 dedupe = exact doc_id across query result sets only; all identity-based dedupe (DOI/PMID, fuzzy) is s2.'

[nit] WHERE: §5.3 header '(before scaling past ~30 papers)' vs its cross-model item ('overnight batch') and M5
ISSUE: The section header implies cross-model verification gates scaling, but the item itself and M5 run it overnight concurrently with the scale-up; G3's conjunction correctly omits it, but the header will make someone ask whether verification must finish first.
FIX: Move cross-model verification out from under the 'before scaling' header or annotate it '(not gate-blocking; runs concurrent with M5)'.

[nit] WHERE: §2 s4 ('patches.yaml applied here with reasons')
ISSUE: patches.yaml has no stated path or author (human-written, presumably), unlike every other artifact in the architecture block.
FIX: Give it a home, e.g. configs/questions/<qid>.patches.yaml, human-authored.

