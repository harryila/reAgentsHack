# Literature Multiverse
## Root Project Context / Hackathon Handoff

> **Normative-status notice (2026-08-15):** This file explains the product thesis, scientific
> motivation, and longer-term design space. It is **not** the implementation contract. The current
> normative specification is
> `docs/superpowers/specs/2026-08-15-literature-multiverse-design.md`, and the executable build order
> is `docs/superpowers/plans/2026-08-15-literature-multiverse-implementation.md`; a locked
> `configs/questions/<qid>.yaml` supplies topic-specific values allowed by that design. If this
> context file conflicts with those authorities, they win. In particular, the example
> schemas in §§6/11, repository tree in §24, coding sequence in §30, and moderator-ranking
> pseudo-code in §31 are historical inputs and must not be copied into production code.

> **Status:** Hackathon build, Track B — “Build a Dataset or Meta-Analysis”  
> **Event:** re:AGENT – End to End Agentic Science, August 15–16, 2026  
> **Core thesis:** Scientific literatures often look contradictory because studies vary along hidden experimental dimensions. Instead of asking an AI to summarize the consensus, build an agentic system that identifies **which experimental choices or contexts make the literature change its answer**.

---

# 0. Read This First

This file is intended to be the root context for the entire project. A coding agent should read this before making architectural decisions.

The project is **not**:

- another literature-search UI;
- a generic RAG assistant over papers;
- an automatic systematic-review writer;
- “ChatGPT, but with citations”;
- a tool that collapses a heterogeneous literature into one consensus sentence;
- a conventional meta-analysis engine that assumes every paper reports directly comparable effect sizes.

The project **is**:

> A system that turns a body of literature into a structured experimental multiverse, measures where the literature disagrees, and discovers the context variables that best explain why different studies reach different conclusions.

The most important product sentence is:

> **Existing literature agents tell you what papers say. Literature Multiverse tells you when the scientific answer changes.**

A second useful sentence:

> **We do not summarize away disagreement. We model it.**

And the hackathon demo sentence:

> **Give us a scientific question. We assemble the studies, extract their experimental contexts and outcomes, and discover the hidden variables that determine when the literature says “yes,” “no,” or “it depends.”**

---

# 1. Why This Fits Track B

The hackathon describes Track B as:

> “What have you always wanted the literature to tell you, but could never read fast enough to find out? Draft the queries, run them across thousands of papers, sharpen and re-run until the results hold the specific pieces you care about. Then the real work: find the pattern in what you assembled that no single paper could show you, and demo that.”

Source: https://luma.com/g6org075

That last sentence is the project.

A normal literature review can tell you:

- Paper A found a positive effect.
- Paper B found no effect.
- Paper C found a negative effect.
- Overall results are “mixed.”

Literature Multiverse asks:

> **Mixed under what conditions?**

Example abstractly:

| Context | Low dose | High dose |
|---|---:|---:|
| Model A | positive | positive |
| Model B | positive | negative |
| Model C | neutral | negative |

The top-line literature looks contradictory. But once dose and model are exposed, the disagreement may have structure.

The useful artifact is not simply the dataset. It is:

1. the structured dataset;
2. the detected heterogeneity;
3. the variables or interactions that explain that heterogeneity;
4. the residual cases that remain contradictory;
5. source-grounded evidence allowing a scientist to inspect every conclusion.

This also matches the event’s broader stated emphasis on infrastructure scientific agents still need: better datasets, sharper tools, and reliable ways to evaluate work.

---

# 2. Intellectual Motivation

## 2.1 The “Garden of Forking Paths” inspiration

James Zou, one of the event hosts, is a coauthor of the July 2026 paper:

**The Agentic Garden of Forking Paths**  
Jiacheng Miao, Jonathan K. Pritchard, James Zou  
https://arxiv.org/abs/2607.01507

Their problem is approximately:

> Same dataset + same research question + different defensible analysis choices → different conclusions.

They show that AI agents can cheaply expose this hidden space of analysis choices and introduce an “Agentic Bootstrap” to sample plausible paths. Their key conceptual move is to stop treating the final reported analysis as the only possible analysis.

**Literature Multiverse is not a reimplementation of that paper.**

Our extension / analogy is:

> Same scientific question + different defensible experimental contexts and study designs → different conclusions.

Instead of a multiverse of **analysis paths over one dataset**, we construct a multiverse of **experimental paths across many papers**.

That distinction is important when explaining the project.

### Their multiverse

```text
                same raw dataset
                      |
          --------------------------
          |            |           |
     analysis A   analysis B  analysis C
          |            |           |
       result +      result -    result 0
```

### Our multiverse

```text
             same scientific question
                      |
        --------------------------------
        |               |              |
     study A          study B        study C
   context X         context Y      context Z
        |               |              |
     result +         result -       result 0
```

The central research question becomes:

> Can an agent discover a low-dimensional set of experimental choices that makes apparently conflicting scientific results predictable?

---

## 2.2 Prior “Serendipity Engine” thinking

A separate line of prior thinking was about autonomous scientific serendipity: instead of retrieving textually similar papers, represent scientific mechanisms and detect non-obvious structural relationships across literatures.

The important lesson from that exploration was **not** “use J-space” or “use a specific latent-space trick.”

The important lesson was:

> Search and semantic similarity are often the wrong abstraction. Scientific discovery may require extracting structured mechanisms, conditions, causal relations, and constraints that are not visible in raw textual proximity.

We considered J-space/Jacobian-lens style representations, but that is the wrong dependency for this hackathon:

- it introduces a difficult interpretability problem before validating the discovery method;
- it is computationally expensive;
- it is not necessary to test whether structured scientific representations expose useful hidden patterns.

For Literature Multiverse, the structured representation is much more concrete:

```text
intervention / exposure
        +
experimental context
        +
population / model
        +
dose / timing / assay
        +
outcome
        ↓
observed result
```

This hackathon project therefore takes the **structural-thinking philosophy** of the Serendipity Engine, but applies it to cross-study disagreement.

Longer-term, embeddings, sparse autoencoders, mechanistic representations, or learned latent spaces could help discover candidate moderators. They should not be MVP dependencies.

---

# 3. The Core Scientific Problem

Scientific disagreement can come from multiple sources:

1. **True biological context dependence**
   - species
   - cell line
   - genotype
   - disease state
   - age
   - sex
   - baseline phenotype

2. **Intervention differences**
   - dose
   - route
   - formulation
   - timing
   - frequency
   - duration
   - combination therapy
   - treatment sequence

3. **Outcome differences**
   - different endpoint definitions
   - different assay technologies
   - different measurement timepoints
   - surrogate vs clinical endpoints

4. **Experimental design differences**
   - in vitro vs in vivo vs human
   - randomized vs observational
   - sample size
   - controls
   - blinding
   - replication
   - study duration

5. **Analysis differences**
   - statistical models
   - exclusion criteria
   - normalization
   - thresholds
   - covariates

6. **Noise / bias**
   - underpowered studies
   - publication bias
   - selective reporting
   - extraction errors
   - duplicated cohorts
   - irreproducible results

The system should **not assume** every disagreement has a clean biological explanation.

The desired output is:

```text
Observed disagreement
        |
        +--> explainable by context variables
        |
        +--> explainable by methodological variables
        |
        +--> residual disagreement / unresolved
```

Residual disagreement is itself useful. Do not force a moderator to “explain” everything.

---

# 4. Main Product Insight

A traditional review asks:

> What does the literature conclude about X?

Literature Multiverse asks:

> What is the conditional answer to X?

Mathematically, instead of estimating only:

\[
P(Y)
\]

we care about:

\[
P(Y \mid X_1, X_2, \ldots, X_k)
\]

where:

- \(Y\) = result / effect direction / effect magnitude;
- \(X_i\) = experimental context variables.

The strongest outcome is not:

> “The literature is 58% positive.”

It is:

> “The global literature is mixed, but 72% of the uncertainty disappears once we condition on model system, dose, and treatment timing.”

That is the “pattern no single paper could show.”

---

# 5. What Counts as a Finding?

## Critical architecture decision

**Do not use one row per paper.**

A paper may contain:

- three experiments;
- multiple doses;
- multiple cell lines;
- several endpoints;
- multiple timepoints;
- positive and negative outcomes in the same paper.

If the system assigns one label to an entire paper, it throws away precisely the context needed to explain disagreement.

### MVP atomic unit: `FindingRow`

One row should correspond to:

> one paper × one experimental comparison × one outcome × one relevant timepoint/context

This can repeat paper metadata.

Example:

| paper | model | dose | duration | outcome | direction |
|---|---|---:|---:|---|---|
| P1 | mouse | 5 mg/kg | 4 wk | tumor volume | - |
| P1 | mouse | 20 mg/kg | 4 wk | tumor volume | + |
| P1 | mouse | 20 mg/kg | 4 wk | survival | 0 |

That is not duplication. Those are three distinct findings.

Longer-term relational schema:

```text
Paper
  └── Study / Experiment
        └── Comparison
              └── Outcome Observation
```

For the hackathon, a denormalized `findings.parquet` or `findings.jsonl` is probably fastest.

---

# 6. Minimum Data Schema

The schema must balance scientific richness against extractability.

Do not begin with 80 fields. Start with a core schema, inspect missingness, then add moderators iteratively.

## 6.1 Core finding fields

```python
FindingRow:
    finding_id: str

    # provenance
    paper_id: str
    title: str
    doi: str | None
    pub_year: int | None
    source: str | None
    peer_review_status: str | None

    # eligibility
    eligible: bool
    exclusion_reason: str | None

    # study/model context
    study_type: str | None
    organism: str | None
    species: str | None
    strain_or_cell_line: str | None
    sex: str | None
    age: str | None
    disease_or_state: str | None

    # intervention/exposure
    intervention: str | None
    comparator: str | None
    dose_value: float | None
    dose_unit: str | None
    route: str | None
    frequency: str | None
    duration_value: float | None
    duration_unit: str | None
    timing_context: str | None
    combination_context: str | None

    # outcome
    outcome_name: str
    outcome_measure: str | None
    assay_or_measurement: str | None
    outcome_timepoint: str | None

    # result
    effect_direction: Literal["positive", "negative", "null", "mixed", "unclear"]
    effect_size_type: str | None
    effect_size: float | None
    effect_size_se: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    statistically_significant: bool | None

    # evidence / grounding
    finding_summary: str
    evidence_quote: str | None
    evidence_section: str | None
    evidence_lines: list[str] | None
    extraction_confidence: float | None

    # quality / dependence
    sample_size: int | None
    independent_replication: bool | None
    dataset_or_cohort_id: str | None
```

## 6.2 `effect_direction` needs domain semantics

“Positive” and “negative” must be defined relative to the research question.

For example:

> “Does intervention X increase outcome Y?”

Then:

- `positive` = X increases Y;
- `negative` = X decreases Y;
- `null` = no detectable effect;
- `mixed` = incompatible directions in the same atomic comparison;
- `unclear` = cannot safely determine.

Do not let the LLM infer whether “positive” is medically good or bad.

---

# 7. Question Configuration

Every literature multiverse should be driven by an explicit config.

Example:

```yaml
question_id: example-question

research_question: >
  Under what experimental conditions does intervention X increase,
  decrease, or have no detectable effect on outcome Y?

target_relation:
  exposure: intervention X
  outcome: outcome Y
  positive_definition: intervention X increases outcome Y
  negative_definition: intervention X decreases outcome Y

eligibility:
  include:
    - primary experimental studies
    - direct measurement of outcome Y
  exclude:
    - reviews
    - editorials
    - purely theoretical work
    - papers that mention X or Y without testing their relationship

initial_moderators:
  - study_type
  - species
  - disease_or_state
  - dose
  - duration
  - route
  - outcome_measure
  - outcome_timepoint

search_queries:
  - '"intervention X" "outcome Y"'
  - '"intervention X" effect "outcome Y"'
  - '"intervention X" increases decreases "outcome Y"'

analysis:
  target: effect_direction
  min_rows_per_stratum: 3
  max_missingness_for_primary_moderator: 0.50
```

Put configs under:

```text
configs/questions/
```

The engine should not hard-code one biological topic.

---

# 8. Paperclip: Current Capabilities We Can Exploit

Current docs: https://paperclip.gxl.ai/docs

Paperclip currently exposes:

- 11M+ full-text papers;
- 150M+ abstracts;
- bioRxiv;
- medRxiv;
- PubMed Central;
- arXiv;
- FDA/regulatory documents;
- ClinicalTrials.gov and international registries;
- UniProt;
- PDB;
- ChEMBL;
- hybrid BM25 + vector search;
- SQL;
- parallel per-paper `map`;
- `reduce`;
- regex/grep;
- paper figures;
- Python SDK;
- an agent/Cursor skill;
- git-like paper repos with claim verification/provenance.

This is why the project should build **on top of Paperclip**, not rebuild retrieval.

## 8.1 Install for Cursor

Recommended current installer:

```bash
curl -fsSL https://paperclip.gxl.ai/install.sh | bash
```

Verify:

```bash
paperclip config
```

Install the Paperclip skill into the project/Cursor:

```bash
paperclip install --dir .
```

There is also:

```bash
npx gxl-paperclip --cursor
```

For scripts / CI:

```bash
export PAPERCLIP_API_KEY="..."
```

The Python package is `gxl_paperclip`.

## 8.2 Fast exploration

```bash
paperclip search "YOUR SCIENTIFIC QUESTION" -n 20
```

Narrow to sources:

```bash
paperclip search --source pmc,biorxiv "YOUR QUERY" -n 50
```

Sort/filter by date:

```bash
paperclip search "YOUR QUERY" --year 2024 --sort date -n 50
```

Vector vs lexical:

```bash
paperclip search --ranking vector "MECHANISTIC QUERY" -n 50
paperclip search --ranking bm25 "EXACT TARGET TERM" -n 50
```

Exact phrase:

```bash
paperclip search -e "EXACT PHRASE"
```

Map a question across results:

```bash
paperclip map --from s_xxxxx \
  "For each eligible primary experiment, what model, intervention, dose, comparator, outcome, and direction of effect were reported?"
```

Paperclip supports strict JSON Schema output on `map`, which is extremely useful here.

## 8.3 Important operational detail

Paperclip’s documentation recommends very small result sets (roughly 3–10) for fast map iteration.

Therefore:

**During schema/prompt development:**
- work on 5–10 papers;
- inspect every extraction;
- revise prompt/schema;
- only then scale.

Do not launch a 500-paper extraction before validating the schema.

---

# 9. Paperclip Extraction Strategy

## 9.1 Search should optimize recall before precision

Use several search formulations and merge/deduplicate.

Possible search families:

1. direct relationship:
   - `"X Y"`
2. mechanism:
   - `"X mechanism Y"`
3. direction:
   - `"X increases Y"`
   - `"X decreases Y"`
4. synonyms:
   - aliases for X
   - aliases for Y
5. study-type specific:
   - human
   - mouse
   - in vitro
   - trial

Paperclip supports `--tag` result accumulation, which can be useful for merged search sets.

## 9.2 Reviews are useful for discovery but not independent evidence

A review can help enumerate:
- synonyms;
- known moderators;
- important papers;
- terminology.

But the core dataset should preferentially contain **primary evidence**, otherwise one original experiment can appear many times through review citations.

Set:

```json
{
  "eligible": false,
  "exclusion_reason": "review_article"
}
```

for non-primary evidence when building the actual finding dataset.

---

# 10. Suggested Extraction Prompt

The extraction model should be conservative.

Key rule:

> If a field is not explicitly supported, return null/unclear. Never infer a dose, species, direction, or statistical result from general background statements.

Prompt skeleton:

```text
You are extracting structured experimental findings for a cross-study
heterogeneity analysis.

RESEARCH QUESTION:
{research_question}

TARGET RELATION:
{target_relation}

For this paper, identify every PRIMARY EXPERIMENTAL FINDING that directly
tests the target relation.

A "finding" is one distinct experimental comparison × relevant outcome ×
timepoint/context. A single paper may contain zero, one, or many findings.

Rules:
1. Do not treat background statements or cited prior work as this paper's findings.
2. Do not treat review summaries as primary evidence.
3. Do not collapse different doses, model systems, outcomes, or timepoints when
   they yield different results.
4. Use null when a context field is not explicitly reported.
5. effect_direction is defined ONLY relative to:
   positive = {positive_definition}
   negative = {negative_definition}
6. "null" means the study reports no detectable difference for this comparison.
7. "unclear" means the paper does not support a safe classification.
8. Include a short verbatim evidence quote and source section/line references
   whenever available.
9. Do not infer that statistical significance implies biological importance.
10. Do not convert a correlational result into a causal claim.

Return only data matching the supplied JSON schema.
```

---

# 11. Example Strict JSON Schema for Paperclip `map`

Start simpler than the full internal schema:

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["paper_id", "eligible", "findings"],
  "properties": {
    "paper_id": {
      "type": ["string", "null"]
    },
    "eligible": {
      "type": "boolean"
    },
    "exclusion_reason": {
      "type": ["string", "null"]
    },
    "findings": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "study_type",
          "species",
          "model",
          "intervention",
          "comparator",
          "dose",
          "duration",
          "outcome",
          "effect_direction",
          "evidence_quote"
        ],
        "properties": {
          "study_type": {"type": ["string", "null"]},
          "species": {"type": ["string", "null"]},
          "model": {"type": ["string", "null"]},
          "intervention": {"type": ["string", "null"]},
          "comparator": {"type": ["string", "null"]},
          "dose": {"type": ["string", "null"]},
          "duration": {"type": ["string", "null"]},
          "outcome": {"type": ["string", "null"]},
          "assay": {"type": ["string", "null"]},
          "timepoint": {"type": ["string", "null"]},
          "effect_direction": {
            "type": "string",
            "enum": ["positive", "negative", "null", "mixed", "unclear"]
          },
          "effect_size": {"type": ["number", "null"]},
          "effect_size_type": {"type": ["string", "null"]},
          "p_value": {"type": ["number", "null"]},
          "sample_size": {"type": ["integer", "null"]},
          "evidence_quote": {"type": ["string", "null"]},
          "evidence_section": {"type": ["string", "null"]},
          "evidence_lines": {
            "type": ["array", "null"],
            "items": {"type": "string"}
          }
        }
      }
    }
  }
}
```

Paperclip's docs note that citation/evidence fields need to be included in the schema if we want them returned.

---

# 12. Agentic Extraction Loop

The system becomes interesting when it is iterative.

## Pass 1 — broad extraction

Extract:
- model;
- intervention;
- outcome;
- effect direction;
- obvious context variables.

## Pass 2 — diagnose disagreement

Compute:

\[
H(Y) = -\sum_y p(y)\log p(y)
\]

where \(Y\) is effect direction.

High entropy = high disagreement.

## Pass 3 — moderator proposal

An agent gets:

- the research question;
- extracted schema;
- missingness profile;
- current outcome distribution;
- a sample of disagreeing studies.

Ask it:

> What experimentally meaningful variables could plausibly distinguish these outcomes that we are not yet extracting?

Candidate output:

```json
[
  {
    "moderator": "treatment timing",
    "reason": "positive studies administer before challenge while negative studies administer after",
    "search_terms": ["pretreatment", "post-treatment", "prior to", "after challenge"]
  }
]
```

## Pass 4 — targeted re-map

Do **not** re-extract everything.

Run a targeted Paperclip map over the same corpus:

```text
Extract treatment timing relative to experimental challenge.
Return one of: before, during, after, unclear.
```

Merge by paper/finding.

## Pass 5 — test explanatory value

Measure whether the new variable actually improves out-of-sample prediction of the result.

If yes:
- keep it;
- surface it.

If no:
- discard or demote it.

## Pass 6 — residual analysis

Find study pairs that remain contradictory despite similar extracted context.

Ask:

> What unmodeled variable distinguishes these studies?

Repeat until:
- improvement stalls;
- extraction budget is reached;
- evidence becomes too sparse.

This is a good place for an “agentic” loop without turning the project into an unbounded autonomous agent.

---

# 13. Quantifying “Why the Literature Disagrees”

A hidden-variable claim should not be based only on an LLM explanation.

We need numerical evidence.

## 13.1 Baseline disagreement

For categorical direction:

```text
positive / negative / null
```

Use Shannon entropy:

\[
H(Y) = - \sum_y p(y)\log p(y)
\]

Normalize:

\[
H_\text{norm}(Y) = H(Y) / \log(K)
\]

where \(K\) is number of outcome categories.

Interpretation:

- 0 → literature is directionally consistent;
- 1 → maximal directional uncertainty.

---

## 13.2 Single-moderator explanatory score

For candidate moderator \(X\):

\[
I(X;Y) = H(Y) - H(Y|X)
\]

Mutual information tells us how much knowing \(X\) reduces uncertainty about the observed result.

A convenient normalized score:

\[
E_X = \frac{H(Y)-H(Y|X)}{H(Y)}
\]

Call this something descriptive such as:

**Disagreement Explained**

Do not overbrand it as a new statistical quantity unless necessary.

Example:

> Model type explains 31% of directional uncertainty.

Caveat: in small samples, raw mutual information can overfit.

Therefore, the actual ranking should preferably use cross-validation.

---

# 14. Preferred Moderator Ranking: Out-of-Sample Predictive Gain

A more defensible test:

### Baseline model

Predict global class probabilities:

\[
P(Y)
\]

### Moderator model

Predict:

\[
P(Y | X)
\]

### Score

Cross-validated change in negative log loss:

\[
\Delta L = L_\text{baseline} - L_\text{moderator}
\]

Positive \(\Delta L\) means the moderator improves prediction on held-out findings.

For multiple variables:

- logistic regression;
- shallow decision tree;
- random forest only as a discovery model;
- gradient boosting only if enough data.

For the demo, prefer **interpretable models**.

Best candidate:
- decision tree depth 2–4;
- one-hot categorical variables;
- binned numeric variables when scientifically sensible.

The tree itself becomes a visualization:

```text
                         STUDY TYPE?
                   / human          \ animal
                DOSE?                  species?
             / low \ high          / mouse \ rat
           positive  null        positive  negative
```

This is literally a “map of the literature multiverse.”

---

# 15. Interaction Discovery

The best findings may not be one variable.

Example:

- dose alone weak;
- species alone weak;
- **dose × species** strong.

Methods, increasing in complexity:

1. contingency tables;
2. pairwise conditional entropy;
3. shallow decision tree;
4. logistic regression with selected interactions;
5. hierarchical model / meta-regression later.

Do not brute-force hundreds of interactions on 40 rows.

Require:
- minimum observations per leaf;
- cross-validated gain;
- stability under bootstrap.

---

# 16. If Numeric Effect Sizes Are Comparable

Some questions support real meta-analysis.

Then add a stronger path:

## Random-effects meta-analysis

Model effect sizes:

\[
\theta_i = \mu + u_i + \epsilon_i
\]

with:
- \(\mu\): overall mean effect;
- \(u_i\): between-study heterogeneity;
- \(\epsilon_i\): within-study uncertainty.

Report:
- pooled effect;
- confidence interval;
- \(I^2\);
- \(\tau^2\).

## Meta-regression

\[
\theta_i =
\beta_0 +
\beta_1 X_{i1} +
\beta_2 X_{i2} + \cdots +
u_i + \epsilon_i
\]

Then ask:

> How much does the moderator reduce \(\tau^2\)?

This is more statistically orthodox than classifying signs.

### But

Do **not** force numeric meta-analysis if:
- effect-size units differ;
- endpoints differ;
- study types differ too radically;
- uncertainty is not reported;
- measurements are not commensurate.

For the hackathon, categorical directional analysis is valid as a **literature-structure analysis**, but should not be presented as a substitute for a proper clinical meta-analysis.

---

# 17. Dependency / Pseudoreplication Problems

This is important.

If one paper contributes 15 findings and another contributes 1, naïvely treating all 16 rows as independent gives the first paper 15× the weight.

Mitigations:

### MVP

For statistics:
- weight each finding by `1 / findings_in_paper`, or
- bootstrap at the **paper level**, not row level.

### Better

Use:
- cluster-robust standard errors;
- hierarchical models with paper random effects.

Likewise, multiple papers can use the same:
- cohort;
- clinical trial;
- public dataset;
- animal dataset.

If detectable, include:

```text
dataset_or_cohort_id
```

and avoid counting republished analyses as independent replication.

---

# 18. Evidence Quality

Do not build a fake universal “study quality score” in one night.

Instead expose interpretable fields:

- study type;
- sample size;
- randomized?
- blinded?
- primary experiment?
- peer-reviewed/preprint?
- independent replication?
- effect size reported?
- uncertainty reported?

Then allow sensitivity analyses.

Example:

> Pattern persists when restricted to peer-reviewed in vivo studies.

That is more defensible than:

> “AI quality score = 0.83.”

---

# 19. Missing Data

Missingness is a first-class property.

For every field compute:

```text
coverage = known_rows / eligible_rows
```

A moderator with:
- huge predictive power;
- 80% missingness;

should not be the headline result.

Dashboard should show:

```text
Moderator        Predictive gain    Coverage
Dose                  high            91%
Species               medium          98%
Timing                high            37%
Assay                  low            82%
```

Rules:

- never convert null into a scientific category like “low dose”;
- “unknown” may be a model category, but do not interpret it scientifically;
- show coverage with every moderator;
- optionally re-map missing high-value fields.

---

# 20. Extraction Validation

Before trusting downstream analysis:

## Human spot-check

Randomly sample 10–20 extracted findings.

For each check:
- is it actually a primary experiment?
- is model correct?
- is intervention correct?
- is comparator correct?
- is outcome correct?
- is direction correct?
- does quote support the row?
- were multiple experiments improperly collapsed?

Record extraction precision.

Even:

> “18/20 randomly audited findings were correctly extracted.”

is a powerful hackathon credibility point.

## Automated consistency tests

Examples:

```python
assert effect_direction in VALID_DIRECTIONS

if effect_size is not None:
    assert effect_size_type is not None

if eligible is False:
    assert len(findings) == 0

if p_value is not None:
    assert 0 <= p_value <= 1
```

Also flag impossible dose/unit combinations.

---

# 21. Contradiction Pairing

A very useful intermediate artifact:

Find pairs of studies with:

- same target relation;
- opposite effect direction;
- high similarity on known context variables.

Define:

\[
D_\text{context}(i,j)
\]

and find:

```text
effect_direction_i != effect_direction_j
AND
context_distance(i,j) is small
```

These are **residual contradictions**.

Why useful:

1. they are excellent demo cards;
2. they tell the moderator-discovery agent where to look;
3. they expose extraction errors;
4. they reveal missing variables;
5. they prevent the system from hiding unresolved disagreement.

---

# 22. Suggested Core Outputs

## 22.1 Literature overview

```text
287 eligible findings
142 positive
81 null
64 negative

Directional disagreement: HIGH
```

## 22.2 “Why papers disagree”

Ranked table:

| Moderator | CV gain | Coverage | Interpretation |
|---|---:|---:|---|
| model system | +0.18 | 96% | largest split |
| dose | +0.13 | 84% | non-linear |
| timing | +0.08 | 61% | pre/post intervention |
| assay | +0.03 | 91% | weak |

## 22.3 Multiverse decision tree

An interpretable tree or Sankey:

```text
all evidence
 ├── human
 │    ├── low dose → mostly positive
 │    └── high dose → mixed
 └── preclinical
      ├── model A → positive
      └── model B → negative
```

## 22.4 Context heatmap

Top two moderators:

```text
                  low dose  medium  high
Model A               +       +      0
Model B               +       0      -
Model C               0       -      -
```

## 22.5 Evidence cards

Click a cell/leaf:

```text
Context:
mouse + high dose + 4 weeks

Evidence:
9 findings
2 positive
1 null
6 negative

Representative papers:
- Paper 1 — source quote
- Paper 2 — source quote
- Paper 3 — source quote
```

## 22.6 Residual contradictions

```text
These studies remain incompatible after conditioning
on all currently modeled variables.
```

That is a feature, not a failure.

---

# 23. The UI

## Recommendation for hackathon speed

Use:

- Python analysis core;
- Streamlit for first working dashboard;
- Plotly for interactive visualizations.

Only move to a separate React/Next.js UI if the core result is already working.

The science is the demo. Do not spend six hours on frontend before finding a real pattern.

## Suggested dashboard layout

### Page header

**Literature Multiverse**

> Ask not “What does the literature say?”  
> Ask “Under what conditions does it say something different?”

### Section 1 — Question

Research question + corpus stats.

### Section 2 — Global disagreement

Stacked bar:
- positive;
- null;
- negative.

### Section 3 — Hidden variables

Ranked moderators.

### Section 4 — Multiverse map

Decision tree / Sankey / heatmap.

### Section 5 — Evidence

Clicking a branch shows source-grounded findings.

### Section 6 — Residual disagreement

Show unresolved contradiction pairs.

### Optional Section 7 — Agent loop

Small trace:

```text
Pass 1: detected high outcome disagreement
Pass 2: model type reduced uncertainty
Pass 3: residual contradictions concentrated by treatment timing
Pass 4: extracted treatment timing
Pass 5: held-out prediction improved 9.2%
```

This makes the system feel agentic without displaying private chain-of-thought.

---

# 24. Project Architecture

Recommended repo:

```text
literature-multiverse/
│
├── PROJECT_CONTEXT.md          # this file
├── README.md                   # concise public-facing README later
├── pyproject.toml
├── .env.example
├── .gitignore
│
├── configs/
│   └── questions/
│       ├── example.yaml
│       └── selected_demo.yaml
│
├── prompts/
│   ├── extraction.md
│   ├── moderator_proposal.md
│   ├── targeted_remap.md
│   └── evidence_audit.md
│
├── schemas/
│   ├── paperclip_extraction.schema.json
│   └── finding.schema.json
│
├── src/
│   └── literature_multiverse/
│       ├── __init__.py
│       ├── config.py
│       ├── models.py
│       ├── paperclip_client.py
│       ├── search.py
│       ├── extract.py
│       ├── normalize.py
│       ├── dedupe.py
│       ├── validate.py
│       ├── disagreement.py
│       ├── moderators.py
│       ├── contradictions.py
│       ├── evidence.py
│       └── pipeline.py
│
├── scripts/
│   ├── 00_smoke_test_paperclip.py
│   ├── 01_search.py
│   ├── 02_extract.py
│   ├── 03_analyze.py
│   └── 04_export_demo.py
│
├── data/
│   ├── raw/
│   ├── extracted/
│   ├── processed/
│   └── cache/
│
├── artifacts/
│   ├── figures/
│   ├── tables/
│   └── demo/
│
├── app/
│   └── streamlit_app.py
│
└── tests/
    ├── test_models.py
    ├── test_normalize.py
    └── test_analysis.py
```

---

# 25. Python Stack

Minimal:

```text
pydantic
pandas
pyarrow
numpy
scipy
scikit-learn
statsmodels
plotly
streamlit
python-dotenv
pyyaml
tenacity
```

Paperclip:

```text
gxl-paperclip
```

Optional:
- `polars` if dataset grows;
- `networkx` for evidence/citation graph;
- `duckdb` for local analytical queries;
- `rapidfuzz` for entity normalization.

For hackathon speed, pandas is enough.

---

# 26. Core Internal Interfaces

## `search.py`

```python
def discover_papers(question_config) -> list[PaperRef]:
    ...
```

Responsibilities:
- run multiple searches;
- merge;
- dedupe by paper ID / DOI;
- save raw IDs and metadata.

## `extract.py`

```python
def extract_findings(paper_refs, config) -> list[FindingRow]:
    ...
```

Responsibilities:
- batch map;
- strict schema;
- cache outputs;
- retry failures;
- flatten one paper → many findings.

## `normalize.py`

```python
def normalize_findings(df, config) -> pd.DataFrame:
    ...
```

Responsibilities:
- unit parsing;
- category normalization;
- synonyms;
- model names;
- direction consistency.

Do not silently convert scientifically distinct values.

## `disagreement.py`

```python
def outcome_entropy(y) -> float:
    ...

def disagreement_report(df) -> dict:
    ...
```

## `moderators.py`

```python
def rank_moderators(
    df,
    target="effect_direction",
    candidate_features=None
) -> pd.DataFrame:
    ...
```

Include:
- coverage;
- support;
- mutual information;
- CV log-loss gain;
- bootstrap stability.

## `contradictions.py`

```python
def find_residual_contradictions(
    df,
    context_features,
    max_distance=...
) -> pd.DataFrame:
    ...
```

## `pipeline.py`

High-level orchestration.

---

# 27. Normalization Is Harder Than It Looks

LLM extraction might produce:

```text
"mouse"
"mice"
"Mus musculus"
"C57BL/6 mice"
```

Do not normalize all of those into the same field.

Prefer hierarchical fields:

```text
species = "Mus musculus"
strain_or_cell_line = "C57BL/6"
```

Similarly:

```text
dose_raw = "10 mg/kg/day"
dose_value = 10
dose_unit = "mg/kg/day"
```

Preserve raw text.

Pattern:

```text
field_raw
field_normalized
```

for high-risk variables.

Never discard the original extracted string.

---

# 28. Topic Selection: Do This Empirically

Do not choose a topic because it sounds cool.

A good demo literature has:

1. enough papers;
2. genuine disagreement;
3. outcomes that can be normalized;
4. several plausible moderators;
5. enough context reported in methods;
6. an intuitive result;
7. biological or clinical relevance;
8. a question understandable to judges in 15 seconds.

## Contradiction-triage procedure

For each candidate topic:

1. search 30–50 papers;
2. map 10 representative papers;
3. extract:
   - eligibility;
   - model;
   - outcome;
   - direction;
   - obvious moderators;
4. score:
   - number of eligible findings;
   - directional entropy;
   - field completeness;
   - context diversity;
   - outcome comparability.

Choose the question with the best combination.

### Candidate topic families to test

These are **starting hypotheses**, not claims that a clean multiverse definitely exists:

#### A. Time-restricted eating / intermittent fasting → metabolic outcomes
Potential moderators:
- early vs late eating window;
- caloric restriction vs isocaloric design;
- baseline metabolic state;
- duration;
- adherence;
- outcome definition.

Pros:
- human relevance;
- many trials;
- understandable.

Risk:
- outcomes are heterogeneous.

#### B. Metformin × exercise adaptation
Potential moderators:
- age;
- training modality;
- metformin dose;
- baseline insulin sensitivity;
- endpoint: VO2max, hypertrophy, insulin sensitivity.

Pros:
- crisp “helps vs blunts adaptation” story.

Risk:
- corpus may be smaller.

#### C. Antioxidant supplementation × exercise adaptations
Potential moderators:
- vitamin/dose;
- training type;
- baseline status;
- endpoint;
- duration.

Pros:
- classic context-dependent question.

#### D. Ketogenic diet × cancer outcomes
Potential moderators:
- cancer type;
- genotype;
- animal/human;
- diet composition;
- concurrent therapy.

Pros:
- large and mechanistically interesting.

Risk:
- extremely heterogeneous; causal claims easy to overstate.

#### E. Rapamycin / mTOR inhibition × aging or immune outcomes
Potential moderators:
- dose;
- intermittent vs continuous;
- age at initiation;
- sex;
- species;
- endpoint.

Pros:
- clear context dimensions.

Risk:
- “outcome” must be scoped tightly.

#### F. One specific intervention with known bidirectional biology
Potential examples:
- a pathway that is protective in one model but harmful in another;
- a drug with dose-dependent reversal;
- an intervention whose effect changes by disease stage.

This can produce the strongest demo if a domain expert at the hackathon suggests one.

---

# 29. A Topic Is Bad If…

Reject or narrow a topic if:

- >50% of papers are reviews;
- most papers do not directly test the relation;
- “positive” means completely different things across papers;
- every paper uses a unique outcome;
- there are fewer than ~20 useful findings;
- one result direction dominates >90%;
- most moderators are unreported;
- one model system accounts for nearly all evidence;
- the interesting contradiction disappears after reading the papers because search retrieval mixed unrelated questions.

---

# 30. First Coding Sequence

A Cursor agent should follow this order.

## Step 1 — initialize project

Create:
- package;
- config loader;
- Pydantic models;
- local directories.

## Step 2 — verify Paperclip

Implement a smoke test:
- search five papers;
- print IDs/titles;
- map one simple field;
- save raw JSON.

Nothing else until this works.

## Step 3 — implement `FindingRow`

Unit tests for:
- required fields;
- effect direction;
- null handling;
- one paper → multiple findings.

## Step 4 — make a tiny extraction run

5–10 papers.

Manually inspect all output.

## Step 5 — normalize

Only normalize fields present in the current topic.

## Step 6 — disagreement analysis

Implement:
- class counts;
- entropy;
- plots.

## Step 7 — moderator ranking

Implement:
- coverage;
- support;
- mutual information;
- CV log-loss gain.

## Step 8 — shallow tree

Create the first visual “multiverse.”

## Step 9 — evidence cards

Keep source paper ID + quote + location.

## Step 10 — scale corpus

Only after output quality looks good.

## Step 11 — targeted re-map

Add one missing moderator discovered from residual conflicts.

## Step 12 — dashboard

Build the demo around the real discovered result.

---

# 31. Suggested Moderator-Ranking Implementation

Pseudo-code:

```python
def rank_moderators(df, target, features):
    baseline = cross_validated_log_loss(
        model=DummyClassifier(strategy="prior"),
        X=np.zeros((len(df), 1)),
        y=df[target],
        groups=df["paper_id"],
    )

    results = []

    for feature in features:
        subset = df[df[feature].notna()].copy()

        if subset[feature].nunique() < 2:
            continue

        if len(subset) < MIN_ROWS:
            continue

        model = make_pipeline(
            OneHotEncoder(handle_unknown="ignore"),
            LogisticRegression(max_iter=1000),
        )

        loss = grouped_cv_log_loss(
            model,
            X=subset[[feature]],
            y=subset[target],
            groups=subset["paper_id"],
        )

        results.append({
            "feature": feature,
            "coverage": len(subset) / len(df),
            "cv_logloss_gain": baseline - loss,
            "n": len(subset),
        })

    return pd.DataFrame(results).sort_values(
        "cv_logloss_gain",
        ascending=False
    )
```

Important:

Use **grouped CV by paper ID** to reduce leakage from multiple findings in one paper.

---

# 32. Bootstrap Stability

A moderator should not headline if it wins only because of three papers.

Bootstrap at paper level:

1. sample papers with replacement;
2. include all findings from selected papers;
3. rerun moderator ranking;
4. record rank/sign/gain.

Report:

```text
Dose ranked in the top 3 moderators in 87% of bootstrap resamples.
```

This is excellent credibility for a hackathon project.

---

# 33. Statistical Safety Rules

Do not claim:

> “X causes the literature disagreement.”

Prefer:

> “X is the strongest observed moderator of study outcomes.”

or:

> “Conditioning on X substantially reduces cross-study disagreement.”

Unless study design supports causal inference.

Likewise:

> “These papers disagree because of X”

is too strong.

Better:

> “Much of the observed disagreement is stratified by X.”

---

# 34. What We Should Compare Against

To demonstrate that the system adds something beyond summary:

## Baseline 1 — Majority vote

```text
55% positive → “overall literature is positive”
```

Show why this hides important subgroups.

## Baseline 2 — LLM consensus summary

Ask an LLM:

> What does the literature say?

Then compare to the multiverse.

The summary may say:

> “Evidence is mixed.”

Our output says:

> “Evidence is mixed globally, but becomes directionally consistent within three major context strata.”

That is the core win.

## Baseline 3 — Single-variable stratification

If our iterative system discovers a more informative variable or interaction, compare.

---

# 35. Demo Narrative

A strong 90-second structure:

## 0–15 sec — Problem

> Scientific reviews collapse a literature into a consensus. But when studies disagree, “mixed evidence” often hides the most important information: the experimental condition under which the answer flips.

## 15–30 sec — Dataset

> We used Paperclip to read [N] papers and extracted [M] individual experimental findings, including model, dose, timing, assay, outcome, and effect direction.

## 30–55 sec — Discovery

Show global distribution:

```text
positive / null / negative
```

> Globally this looks contradictory.

Click multiverse tree:

> But once we condition on [top moderator] and [second moderator], the literature separates into distinct regimes.

## 55–70 sec — Evidence

Click a leaf.

> Every point is traceable to the original paper and supporting passage.

## 70–82 sec — Agentic loop

> The system did not need us to pre-specify every variable. It found residual contradictions, proposed an additional moderator, re-read the corpus for that field, and retained it only when it improved held-out prediction.

## 82–90 sec — End

> Literature agents answer “what do papers say?” Literature Multiverse answers “when does the scientific answer change?”

---

# 36. What Would Make the Demo Exceptional

Any one of these:

### A. A real reversal

> Under condition A, 80% of studies are positive.  
> Under condition B, 75% are negative.

### B. A hidden threshold

Dose or duration reveals a sign flip.

### C. A methodological artifact

The conclusion changes primarily by assay or outcome definition.

### D. A model-system mismatch

Preclinical models strongly support an effect that human studies do not.

### E. A previously unappreciated interaction

Neither X nor Y alone explains much, but X × Y does.

### F. Residual contradiction with a clear next experiment

> The literature becomes consistent everywhere except this sparsely studied condition.

Then Literature Multiverse can suggest:

> “This is the highest-information experiment to run next.”

That last feature begins to bridge Track B toward an AI Scientist without leaving Track B.

---

# 37. Optional: Next-Experiment Selection

Once the literature is represented as context → outcome, identify regions with:

- high uncertainty;
- high scientific relevance;
- low evidence count;
- disagreement between neighboring contexts.

Define an informal acquisition score:

\[
A(x) =
\text{uncertainty}(x)
\times
\text{importance}(x)
\times
\text{evidence scarcity}(x)
\]

Then:

> “The literature is missing evidence in exactly this condition.”

This is a powerful post-MVP extension.

Do not implement before the core multiverse works.

---

# 38. Optional: Temporal Backtesting

A future research version could use publication time.

Train/extract only literature before year \(T\).

Ask:

- what moderators appear important?
- what context regions are uncertain?
- what future experiments would the system prioritize?

Then reveal post-\(T\) papers.

Did later work:
- confirm the predicted moderator?
- fill the high-information region?
- resolve the contradiction?

This connects Literature Multiverse back to the earlier Serendipity Engine goal of evaluating whether a discovery system could have surfaced useful structure before humans explicitly reported it.

Not MVP.

---

# 39. Optional: Mechanism Layer

Current project:

```text
context → outcome
```

Longer-term:

```text
context
   ↓
mechanism
   ↓
outcome
```

Then disagreements could be explained mechanistically:

> Condition A activates pathway P; condition B activates compensatory pathway Q.

This would make the system more scientifically explanatory.

Potential fields:

```text
target
pathway
mediator
direction_of_regulation
downstream_state
phenotype
```

Again: after the simple system works.

---

# 40. Optional: Cross-Domain Serendipity

The eventual bridge to the separate Serendipity Engine:

1. Literature Multiverse learns motifs like:
   - dose-dependent sign reversal;
   - compensatory pathway activation;
   - threshold behavior;
   - biphasic response;
   - context-specific failure.

2. Search unrelated literatures for the same structural motif.

3. Suggest:
   > “This field may exhibit the same hidden regime structure.”

This is much more grounded than comparing papers by generic embeddings.

---

# 41. Major Failure Modes

## Failure mode 1 — Search-result bias

If retrieval only finds famous positive papers, analysis is meaningless.

Mitigation:
- multiple query formulations;
- explicit negative/null terms;
- keyword + vector search;
- broad source coverage;
- inspect retrieval distribution.

## Failure mode 2 — Conclusion-section bias

LLM extracts what authors say rather than what experiment shows.

Mitigation:
- require finding-level evidence;
- prioritize Results/Methods;
- distinguish `author_conclusion` from `measured_result`.

## Failure mode 3 — Outcome collapse

Different outcomes treated as one variable.

Mitigation:
- explicit outcome taxonomy;
- only compare commensurate endpoints;
- stratify outcome family.

## Failure mode 4 — Paper-level collapse

Already covered: one paper may have multiple findings.

## Failure mode 5 — Moderator hallucination

Agent invents an explanation.

Mitigation:
- candidate moderators must be extracted from papers;
- must improve quantitative held-out analysis;
- evidence coverage shown.

## Failure mode 6 — Small-stratum storytelling

A leaf with `n=2` looks dramatic.

Mitigation:
- minimum leaf/support thresholds;
- display N;
- bootstrap stability.

## Failure mode 7 — Publication bias

The indexed literature is not the universe of conducted experiments.

Mitigation:
- say this clearly;
- optionally include trial registries;
- compare registered vs published evidence later.

## Failure mode 8 — Duplicate evidence

Same trial/cohort appears in multiple papers.

Mitigation:
- DOI/PMID dedupe;
- detect trial IDs;
- cohort IDs;
- cluster/weight.

## Failure mode 9 — Confusing predictive with causal

A moderator may correlate with another hidden variable.

Mitigation:
- careful language;
- interaction/sensitivity analysis;
- residual contradiction display.

## Failure mode 10 — Building the UI before the result

The most dangerous hackathon failure.

The priority is:

```text
real pattern > beautiful interface
```

---

# 42. Success Criteria

## Minimum viable success

- 20+ eligible findings;
- provenance;
- visible directional disagreement;
- at least one context variable;
- working stratification visualization.

## Good hackathon success

- 50–200+ findings;
- extraction spot-check;
- several moderators;
- grouped cross-validation;
- one clear hidden regime;
- clickable evidence;
- residual contradictions.

## Excellent

- agent proposes and re-extracts a new moderator;
- new moderator improves held-out prediction;
- bootstrap-stable result;
- pattern is scientifically non-obvious;
- result survives reasonable sensitivity analyses.

---

# 43. Priority Order Under Time Pressure

If things go wrong, cut in this order:

1. cross-domain serendipity;
2. next-experiment planner;
3. fancy frontend;
4. numeric meta-analysis;
5. autonomous multi-iteration moderator loop;
6. complex quality scoring;
7. mechanism extraction.

Do **not** cut:

1. finding-level schema;
2. source grounding;
3. actual disagreement analysis;
4. at least one quantitative moderator test;
5. evidence display;
6. one interpretable demo result.

---

# 44. Recommended Hackathon Milestones

## Milestone 1 — Infrastructure works
- Paperclip installed/authenticated;
- Cursor skill installed;
- search works;
- map works.

## Milestone 2 — One topic selected
- search counts;
- 10-paper extraction;
- disagreement exists;
- context fields are extractable.

## Milestone 3 — Dataset v1
- 30–100 papers;
- flattened findings;
- provenance;
- normalized core moderators.

## Milestone 4 — Scientific result
- global disagreement;
- moderator ranking;
- tree/heatmap;
- one striking conditional pattern.

## Milestone 5 — Trust
- 10–20 finding audit;
- source evidence;
- bootstrap / grouped CV.

## Milestone 6 — Demo
- dashboard;
- crisp narrative;
- no dead-end buttons;
- cached results so the demo does not depend on live extraction.

---

# 45. Cache Everything

Hackathon demo should not depend on live Paperclip/API calls.

Save:

```text
data/raw/search_results/
data/raw/map_results/
data/extracted/findings.jsonl
data/processed/findings.parquet
artifacts/demo/
```

Every pipeline step should be restartable.

Use content/config hashes if easy.

At minimum:

```text
question_id
search_result_id
map_result_id
timestamp
schema_version
prompt_version
```

---

# 46. Provenance Metadata

Every final displayed claim should be reconstructible.

Store:

```python
Provenance:
    paperclip_document_id
    paperclip_search_result_id
    paperclip_map_result_id
    paper_id
    doi
    title
    evidence_quote
    evidence_section
    evidence_lines
    extraction_prompt_version
    extraction_schema_version
```

This is especially important because the event emphasizes trustworthy scientific outputs.

---

# 47. Paperclip Repos Could Be Useful

Paperclip supports git-like paper repositories where claims can be added and commits verify them against source text.

Possible use:

```bash
paperclip repo init literature-multiverse "Evidence corpus for selected question"
```

Then add the strongest demo claims with source lines and commit them.

This is not required for the analysis pipeline, but it could make the demo’s provenance story especially strong:

> “The data point came from an LLM extraction, and the claim was rechecked against the source before entering the evidence repo.”

Do not spend too long integrating this unless it works smoothly.

---

# 48. ClinicalTrials.gov / Registry Opportunity

If the selected topic is clinical, Paperclip has trial registries.

That creates an interesting optional analysis:

```text
registered trials
       vs
published papers
```

Questions:
- Are null trials less likely to appear as publications?
- Do registered endpoints differ from reported endpoints?
- Does the observed multiverse change when unpublished/registry evidence is included?

Potentially extremely strong, but only if the topic makes this easy.

---

# 49. Naming / Messaging

Current name:

# **Literature Multiverse**

Strong because:
- connects to multiverse analysis;
- intuitive;
- visual;
- aligns with “many possible scientific answers depending on path/context.”

Possible taglines:

1. **When does the literature change its mind?**
2. **Find the hidden variables behind scientific disagreement.**
3. **The answer isn’t “mixed.” It’s conditional.**
4. **Map the experimental conditions under which scientific conclusions flip.**
5. **From conflicting papers to conditional scientific laws.**

Recommended:

> **Literature Multiverse — When does the literature change its mind?**

---

# 50. Public README Later

Do not make this giant context file the public README.

Eventually create a concise README with:

1. one-liner;
2. demo GIF/screenshot;
3. what it does;
4. architecture;
5. install;
6. example;
7. limitations;
8. sources.

This file is internal project memory / agent context.

---

# 51. Scientific References / Conceptual Background

### Agentic Garden of Forking Paths
Miao J, Pritchard JK, Zou J. 2026.  
https://arxiv.org/abs/2607.01507

Key relevance:
- exposes hidden variability from defensible analysis choices;
- motivates thinking in distributions over scientific paths rather than one reported path;
- introduces Agentic Bootstrap and m-value for the one-dataset analysis setting.

### Increasing Transparency Through a Multiverse Analysis
Steegen S, Tuerlinckx F, Gelman A, Vanpaemel W. 2016.  
DOI: 10.1177/1745691616658637  
https://pubmed.ncbi.nlm.nih.gov/27694465/

Key relevance:
- formal multiverse-analysis framing;
- evaluate conclusions across reasonable data-processing decisions rather than one arbitrary path.

### Many Analysts, One Data Set
Silberzahn R et al. 2018.  
DOI: 10.1177/2515245917747646

Key relevance:
- independent teams can reach different results from the same data/question.

### Variability in the Analysis of a Single Neuroimaging Dataset by Many Teams
Botvinik-Nezer R et al. 2020. Nature 582, 84–88.  
DOI: 10.1038/s41586-020-2314-9

Key relevance:
- concrete demonstration of analysis-path heterogeneity.

### Paperclip
Current docs:  
https://paperclip.gxl.ai/docs

### Hackathon
re:AGENT – End to End Agentic Science  
https://luma.com/g6org075

---

# 52. Open Research Questions

These are not required for the hackathon, but they define where the project could go.

1. Can cross-study heterogeneity be represented as a learnable experimental state space?
2. Can an agent autonomously discover missing moderator variables?
3. Can hidden-variable discovery be evaluated retrospectively?
4. Can the system distinguish biological context dependence from methodological artifacts?
5. Can contradiction pairs predict replication failures?
6. Can temporal backtesting identify moderators before review articles synthesize them?
7. Can the multiverse identify the most informative next experiment?
8. Can mechanism extraction turn predictive moderators into explanatory models?
9. Can the same framework work outside biology:
   - economics;
   - psychology;
   - ML benchmarks;
   - materials science;
   - climate science?
10. Can “literature robustness” be quantified without pretending papers are IID samples?
11. How should publication bias and duplicated cohorts be incorporated?
12. Can representations from mechanistic graphs, embeddings, SAEs, or other latent-space methods improve moderator discovery?

---

# 53. Core Philosophical Principle

Do not force science into:

```text
TRUE
FALSE
```

Many real findings are:

```text
TRUE under X
FALSE under Y
UNKNOWN under Z
```

The literature already contains those branches, but they are scattered across papers.

The job of Literature Multiverse is to reconstruct the branching structure.

---

# 54. Cursor Bootstrap Instruction

A coding agent opening this repository should be given this instruction:

```text
Read PROJECT_CONTEXT.md completely before modifying the repository.

We are building the Literature Multiverse hackathon project described there.
Do not begin by building a polished UI.

First:
1. inspect the repo,
2. verify Paperclip is installed and callable,
3. create the minimal Python package / schemas / config structure,
4. implement a Paperclip smoke test,
5. implement the FindingRow Pydantic model,
6. create one example question config,
7. implement a 5–10 paper extraction path with strict structured output,
8. persist raw + flattened results,
9. implement global outcome counts/entropy,
10. implement a first moderator ranking with grouped CV by paper.

Keep every step runnable and cached.
Prefer simple, testable functions over a large agent framework.
Preserve source provenance for every extracted finding.
Do not silently invent missing scientific fields.
Do not treat multiple findings from one paper as independent papers.
Do not make causal claims from predictive moderators.
```

---

# 55. Immediate Decision Checklist

Before serious scaling, answer these in the repo:

- [ ] What exact scientific question are we using?
- [ ] What does “positive” mean?
- [ ] What does “negative” mean?
- [ ] What studies are eligible?
- [ ] What is the atomic outcome unit?
- [ ] What are the first 5–10 candidate moderators?
- [ ] Can Paperclip retrieve at least ~30 useful studies/findings?
- [ ] Do we observe real disagreement?
- [ ] Can the relevant methods/context fields be extracted?
- [ ] Is there at least one intuitive visualization?
- [ ] Are representative rows grounded to source passages?
- [ ] What is the one sentence we want judges to remember?

Recommended sentence:

> **The literature wasn’t actually inconsistent — it was conditional. We built the system that found the conditions.**

---

# 56. Final North Star

If the project works, a scientist should be able to ask:

> “Does X affect Y?”

and receive something much more useful than a summary:

```text
Overall literature:
    mixed

But:

    in context A:
        strongly positive

    in context B:
        mostly null

    in context C:
        predominantly negative

The two variables most associated with the change are:
    dose
    model system

These patterns are supported by:
    N findings
    M independent papers

Here are the exact source passages.

These K studies remain contradictory after conditioning
on known variables.

The least-studied, highest-uncertainty regime is:
    context D.
```

That is the product.

That is also the research idea.

And for this hackathon, everything else is secondary to producing **one scientifically credible example where the system converts “mixed literature” into a clear conditional structure that no single paper exposes.**
