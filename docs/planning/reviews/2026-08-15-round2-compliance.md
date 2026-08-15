VERDICT: pass-with-fixes

[should-fix] WHERE: v2 §5.2 Gate G2 floors (line 260-261)
ISSUE: B6 sets the G2 floors as a conjunction ('probe entropy ≥0.4 on ≥1 topic AND est. ≥40 usable primaries; else escalation ladder') — the ladder fires when EITHER floor fails. v2 writes 'Both fail → escalation ladder', which as written triggers escalation only when both floors fail simultaneously; e.g. entropy 0.5 but only 30 usable primaries would pass through with no branch.
FIX: Change 'Both fail → escalation ladder' to 'Either floor fails → escalation ladder' (or 'floors not met →'), matching B6's else-clause semantics.

[should-fix] WHERE: v2 §7 Pattern-failure branch + M4/M4.5 rows (lines 331-332, 345-348)
ISSUE: B6's second written tripwire is 'M4 fails → variant B, cut remap, reclaim the hour for evidence cards.' v2 has the variant-B branch and pre-scripting, but drops 'cut remap' — as written, the M4.5 agent-loop/remap hour (10:30–11:30 PM) still runs even when no moderator passes permutation, and the reclaimed hour for evidence cards is never granted.
FIX: Add to the pattern-failure branch: on M4 failure, cut the M4.5 remap iteration and reassign that hour to contradiction/evidence cards.

[should-fix] WHERE: v2 D10 + §5.2 (lines 158-164, 251-263)
ISSUE: A16 conditions the C+B merge rung: 'exercised only if C's confirmed counts come in <~60 AND A disappoints.' v2 keeps the merge as the first escalation rung but drops the <~60 confirmed-count threshold entirely, leaving the merge trigger tied only to the generic G2 floor failure.
FIX: Restore the explicit trigger in D10 or §5.2: merge rung fires only if C's confirmed counts < ~60 AND A disappoints.

[should-fix] WHERE: v2 §5.3 Gate G3 (line 277-278)
ISSUE: A19b makes G3 a conjunction of 'human audit ≥17/20 AND anchors match AND cross-model agreement ≥ threshold.' v2's G3 reads 'audit ≥17/20 AND anchors match AND paper-bootstrap entropy interval overlaps ≥0.4' — the entropy clause (A12) replaced the cross-model clause instead of both applying. The cross-model batch itself is present (§5.3) but no longer gates anything. (Note: the decision has internal timing tension — the batch runs overnight, after G3 — so if the substitution was deliberate it should be stated as a deviation.)
FIX: Either add 'cross-model agreement ≥ threshold' to the G3 conjunction (e.g. run the check on the v1 subset at G3 time, full corpus overnight), or document the timing-driven deviation explicitly.

[should-fix] WHERE: v2 §2 s4 / §4 data contracts (A20d)
ISSUE: A20d specifies 'explicit Int64 nullable casts' as part of the parquet/normalization contract. This mechanism (preventing pandas silently coercing nullable int columns like sample_size to float) appears nowhere in v2 — every other A20d element (single source of truth, patches.yaml, manifest.json, schema_version assert, created_at footer) landed.
FIX: Add 'explicit Int64 nullable casts' to the s4 normalize description or §4.1 contract notes.

[should-fix] WHERE: v2 D3 + §7 milestone table (lines 111-119, 322-336)
ISSUE: B5 says 'Budget 45 min pre-11 PM for a checkpointed runner.' v2 keeps the runner spec and the not-ready-by-11-PM fallback, but schedules no build slot for the runner anywhere in the milestone table — the exact 'schedules deliverables but never schedules writing the code' failure the adjudicator called out. M4 (≤10:30) and M4.5 (10:30–11:30) leave no time allocated to build it.
FIX: Add the ~45-min runner-build block to the milestone table (or D3) somewhere before 11 PM.

[should-fix] WHERE: v2 §4.1 heading (line 170)
ISSUE: Heading claims 'dieted to ~15 extraction fields (A15)' but the extraction-field list contains 24 fields (study_type through confidence). All six named A15 cuts landed and the two dose fields were added, so the substance is right, but the '~15' number the decision set is not actually met and the label misstates the delivered count.
FIX: Either trim further toward ~15 or correct the heading to the real count so the diet claim matches the list.

[nit] WHERE: v2 D7 trace.json enumeration (lines 140-142)
ISSUE: A11's trace.json field list includes 'input pairs' (the residual pairs the agent saw); v2's enumeration logs proposal verbatim, approval, result IDs, before/after gain, keep/discard, timestamps — but omits input pairs, weakening the 'render what the agent saw' honesty intent.
FIX: Add 'input residual pairs' to the trace.json logged-fields list in D7.

[nit] WHERE: v2 D1 (lines 105-107) and §5.1 auth line
ISSUE: B2 names the env var PAPERCLIP_API_KEY ('run everything on PAPERCLIP_API_KEY'). v2 keeps the full mechanism (durable key minted at paperclip.gxl.ai/keys, never the OAuth session token) but never names the env var.
FIX: Name PAPERCLIP_API_KEY in D1 or M0 alongside ANTHROPIC_API_KEY.

[nit] WHERE: v2 §8 duplicate-cohorts row + D4 (lines 122-126, 365)
ISSUE: A17's G3 same-cohort hand-pass is present ('over headline-leaf papers') but its matching criteria — author overlap + year±2 + identical n/intervention → shared dataset_or_cohort_id — are not spelled out anywhere in v2.
FIX: Add the three matching criteria to the G3 hand-pass description in D4, §5.3, or the risk row.

