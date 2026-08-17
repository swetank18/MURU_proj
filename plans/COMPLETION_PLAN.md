# MURU-BENCH — Completion Plan v1.0

## Progress log

**2026-08-17 — the post-hoc correction is validated; P2's rule-decidable half is done; the
provider deleted most of the panel.**

*The validation.* The paper rests on `unit_accounting`, a rule applied after the fact to
data we already had, so the check that counts is whether a prompt that states the
convention lands where the rule predicts. Registered three outcomes in
`compare_prompt_versions.py` before the answers arrived; got the first one. On the two
models that follow the instruction, v2 is indistinguishable from the v1 *corrected*
scoring and differs from the raw one, and **corroborated unit mismatches go 5→0 and 7→0**.

| model | n | v1 raw | v1 unit | v2 | McNemar vs v1-unit |
|---|---|---|---|---|---|
| GPT-OSS-120B | 100 | 92.0% | **97.0%** | **96.0%** | p = 1.000 (+1/−2) |
| Llama-3.3-70B | 73 | 76.7% | **86.3%** | **84.9%** | p = 1.000 (+2/−3) |
| Llama-3.1-8B | 94 | 47.9% | 50.0% | 44.7% | p = 0.458 (+12/−17) |

The 8B is the caveat, not a counterexample: 29 of 94 pairs flip, so it is near-unstable
under re-prompting and contributes no evidence either way — and it is the only row still
emitting unit mismatches *after* being told the convention. Paper §6.3.

*Endpoint attrition is now a first-class fact.* Groq withdrew `llama-3.3-70b-versatile`
and `llama-3.1-8b-instant` **while the replication was running** (that is why two arms are
73 and 94, not 100). With the two withdrawn in August, **four of five panel endpoints are
gone in fourteen weeks and only `gpt-oss-120b` survives**; re-querying reproduces one row
of five. Rewrote the reproducibility section around the consequence: the committed
archives are the artefact, and a substituted endpoint is a new row rather than a
reproduction. This constrains P1a, P1b and P3 — see the warning added to P1a.

*P2's mechanical half.* Read the corpus, wrote the codebook afterwards, and split
detection: 5 codes decided by written-down rules, 7 left to a judgment pass that returns
nothing rather than guessing. Of 361 raw errors, 86 are unit mismatches credited correct,
leaving 275; rules name 46 (17%) and **229 (83%) are reported as uncoded**. Two results
worth keeping: at the clean end of the panel a large share of what remains is *reporting*
rather than arithmetic (4 of 9 for GPT-OSS, 6 of 17 for Qwen, against 3 of 39 for
Llama-3.3), and **21% of surviving errors are wrong point estimates whose own interval
contains the truth** (31% Llama-3.3, 33% Scout, 6% Qwen) — the sharpest argument in the
paper for scoring more than a point estimate, and it cuts both ways. Paper §7.2.

*Also:* found and fixed a resume bug that would have merged v1 answers into an archive
stamped `prompt_version: 2`. 105 tests. PDF 37pp. Pushed — `origin/main` at `b7b3308`,
which also carried the five commits that had been sitting unpushed since `d220f64`.

**2026-08-16 (2) — P2 groundwork found a scoring artifact that invalidated the headline.**
Step 6 of the old 72-hour list said to read the wrong answers instead of guessing at a
codebook. The first thing they said is that a quarter of them are not wrong.

The v1 prompt asked for "a single number". Many stems are denominated in $K or in percent
and the ground truth is stored in that unit, so a model computing $163,195 against a
ground truth of 164.7 was marked wrong. **86 of 348 recorded errors (24.7%) are correct
answers in an admissible unit.** `evaluation/unit_accounting.py` credits a rescale only
when the model's *own* interval lands on the ground-truth interval under the same factor,
because a bare point estimate that rescales into a wide interval can be a genuine
1000×-too-large answer (MURU-3022 is exactly that). Corrected accuracies are lower bounds:
30 further errors rescale in without corroboration and are not credited.

**Three of four empirical claims did not survive; the user chose to make unit-aware the
primary accounting throughout.**

| claim | before | after |
|---|---|---|
| accuracy ↔ ECE coupling | ρ = −0.90 | **ρ = −0.10** (p = 0.95) |
| Decision-Under-Uncertainty hole | 22.6–64.2% | 76.3–88.7% |
| "coverage bought with width" | p90 width 6×–998× | 1.0×–14.7×, only the 8B hedges |
| discrimination (survives, strengthens) | AUROC 0.465–0.631 | **0.439–0.575**, leader below chance |

The new headline is the null: **accuracy and calibration-in-level are separable**, and the
metrics that appear to track capability (OvConf, Brier — both ρ = −1.00) are functions of
the error rate rather than of calibration shape. That is a better argument for the
benchmark existing than the coupling would have been, and it is the P5 incremental-validity
case arriving early. Panel accuracies are now 43.9 / 87.0 / 88.3 / 93.1 / 97.0.

Root cause fixed: `run_eval.py` states the unit convention and stamps `PROMPT_VERSION` (the
panel is v1; two of five endpoints are withdrawn so it cannot be re-collected under v2 —
a clean-prompt replication on the three survivors is the top validation item). Paper
rewritten throughout (abstract, §6 with a new unit-accounting section + table, coupling
section, metric validity, per-difficulty/category, discussion, conclusion, two new
limitations, reproducibility), README rewritten, both zips rebuilt, PDF 35pp.
`evaluation/error_extract.py` builds the failure-coding corpus (361 errors, 346 readable)
and now excludes corroborated unit mismatches from the coding sample. 83 tests.

**P2 proper (codebook, κ, judge validation) is still not started** — the corpus and the
sampling instrument are ready, and the sample is much cleaner for it.

**2026-08-16 — P4 closed: the overconfidence cut is formalised and swept.** Again pure
re-analysis of the committed archives. `evaluation/overconfidence.py` states the cut
(τ = 0.7, strict `>`) as a reporting choice and sweeps it over {0.5 … 0.99} under both tie
conventions, alongside two cut-free companions. **P4 is now complete.**

- **The comparative claim survives the cut; the absolute level does not.** Ordering is
  identical to τ = 0.7 at τ = 0.6 and 0.8; the only departure in the informative range is
  one adjacent swap at τ = 0.5 between two models 0.1 pp apart at the canonical cut. The
  weakest/strongest ratio is ≥ 7.5× at *every* cut tested.
- **Verbalised confidence is a point mass, and that breaks high cuts.** The single value
  0.95 carries 22.9–70.0% of each model's answers, so a strict cut there discards each
  model's modal block and the ordering collapses (rank corr +0.36) — while the inclusive
  convention at the same cut restores it exactly (+1.00). τ ≥ 0.9 is marked degenerate
  (the leader has 7 confident errors left at 0.9, 3 at 0.95). Generalisable lesson for
  calibration reporting, not a MURU quirk.
- **New finding — the eightfold collapse is the accuracy factor, not metacognition.**
  OvConf factors as P(wrong) × P(confident | wrong). The second term is 79.1–95.7% across
  the panel with no trend in capability (ρ = −0.60, p = 0.35): the leader states high
  confidence on 19 of its 22 errors. This is the same conclusion as the P4 resolution/AUROC
  result, reached independently from the safety metric's own decomposition — and it is the
  reading the paper now gives the metric, replacing an implied metacognitive one.
- **The threshold-free statistic is cleaner than the thresholded one.** Mean overconfidence
  E[max(0, c − y)] is perfectly monotone in accuracy (ρ = −1.00, exact permutation
  p = 0.017) where the rate gives ρ = −0.90 — the near-tie was introduced by the cut.

Shipped: `evaluation/overconfidence.py`, 18 tests (69 total), a sixth generated LaTeX table
(`real_llm_overconfidence.tex`), paper §Metric Validity + Discussion updated, README
leaderboard section added. The canonical τ = 0.7 numbers are unchanged — every previously
published figure still holds.

**2026-08-15 — P0 (parse accounting) and P4 (metric hardening) shipped.** Both were pure
re-analysis of the committed archives: no new API calls, no new runs. What landed:

- `evaluation/parse_status.py` — 10-status taxonomy separating model-side failures
  (truncated / missing_field / format_variant / no_schema / refused) from provider-side
  ones (endpoint 404, rate limit, timeout, …). Every metric now computed twice.
- **The Llama-3.1-8B mystery is solved.** Its 12 missing items are 11 token-budget
  truncations mid-derivation plus 1 empty `POINT_ESTIMATE` field — all model-side. Its
  row is 301/301 attempted, not 289/301 partial. Same for Llama-4-Scout (1 off-schema
  answer). **Only Qwen3-32B is genuinely partial** (56 provider 404s), so the
  partial-coverage-row problem is down to one row with a documented cause.
- **Strict vs lenient gap is at most 1.7 pp anywhere in the panel** → the leaderboard is
  not covertly measuring format compliance. Disclosed in the paper either way.
- `evaluation/calibration_metrics.py` — Winkler interval score (normalised by GT width),
  relative-width distribution + hedge rate, equal-mass/equal-width ECE across 4 bin
  counts, debiased ECE, Brier + Murphy decomposition, AUROC.
- `results/schema.json` + `evaluation/baselines/MANIFEST.json` (SHA-256 per archive) +
  `make reanalyze`. Simulated tiers moved out of the README leaderboard into a separate
  harness-validation table. 51 tests passing.

**Two findings came out of P4 that change the paper:**

1. **Verbalized confidence has near-zero resolution.** Murphy resolution ≤ 0.010 against
   outcome uncertainty 0.154; AUROC 0.465–0.631, i.e. below chance to weak, and it does
   *not* improve with capability. Calibration-in-level improves with capability;
   discrimination does not. This is a second, orthogonal answer to "failing on what" (O2)
   and it is the strongest negative result in the paper.
2. **The ρ = −0.90 rank inversion was an estimator artifact.** Raw ECE's finite-n floor is
   model-dependent (0.036 Qwen vs 0.024 Llama-3.3-70B). Debiased ECE *and* Brier are both
   monotone in accuracy → ρ = −1.00. Reported as a metric correction on the same 5 runs,
   not as new evidence; §6's warning about n=5 still stands unchanged.

Also worth noting against §6: coverage is partly bought with width (10–19% of every
model's intervals are >10× the defensible width), which Acc@CI could not see. That is an
independent reason the headline should lean on the interval score and ECE rather than
Acc@CI alone — which is the same conclusion P5 would reach if MURU turns out to be
redundant on point accuracy.

**Still open, in priority order:** P1b human baseline (the blocking one), P2 failure
taxonomy, P3 A1/A3/A6 ablations, P1a panel expansion, P5, P6. Nothing in those has been
started. P0 is complete bar the `v1.0` dataset tag; P4 is complete.

---

**Status as of 2026-08-15:** dataset + harness are done and defensible. The *evidence* layer is not.
**Driving deadline:** ICLR 2027 — abstract **2026-09-18 AOE**, paper **2026-09-25 AOE**. ~5.5 weeks.
**Repo:** github.com/swetank18/MURU_proj · **DOI:** 10.5281/zenodo.20036750

---

## 0. Diagnosis — what the criticism actually means

The critic said: *"more substantial proof that models are failing, on what, more accreditation, more detail."*

Unpacked, that is four separate objections, and only one of them is about sample size:

| # | Objection | Current state | Severity |
|---|---|---|---|
| **O1** | **"Failing relative to what?"** | No human baseline. "Expert (sim.)" is a *simulator*, not a person. A benchmark cannot claim models fail without a measured ceiling. | **Blocking** |
| **O2** | **"Failing on what?"** | Only aggregate metrics. No error taxonomy, no mechanism, no worked failure examples. A low number is not a finding. | **Blocking** |
| **O3** | **"Is the failure real or an artifact of your setup?"** | Single prompt, single sample, single temperature. Parse failures possibly conflated with wrong answers (n=276 vs 300 on Llama-3.1-8B). Zero prompt ablations. | **Blocking** |
| **O4** | **"Is the panel big enough to support the claim?"** | 2 fully-covered real models + 3 partial + 4 simulated. The headline ρ = −0.90 is computed over ~5 points, most simulated. | **High** |

Plus two that reviewers will raise even though the critic didn't:

| # | Objection | Current state |
|---|---|---|
| **O5** | **"Why does this benchmark need to exist?"** — no incremental-validity evidence over GSM8K/MATH. | Absent |
| **O6** | **Metric validity** — Acc@CI rewards wide intervals; ECE is binning-dependent and biased at small n. | Only coverage-side metrics reported |

**Honest read:** the dataset is the strong part. The empirical section is currently the weak part, and it is the part a benchmark paper is judged on. The good news is that every one of O1–O6 is fixable at near-zero cost in the time available. The bad news is O4 may *invalidate the current headline finding* — see §6.

---

## 1. Phase plan

Six phases. P1–P3 are the load-bearing ones; if time runs out, ship P1–P3 + P6 and cut P4–P5 to "limitations."

```
Week 1 (Aug 15–21)   P0 hygiene + P1 panel expansion (launch, runs in background)
Week 2 (Aug 22–28)   P2 failure taxonomy  + P3 robustness ablations
Week 3 (Aug 29–Sep 4) P1 human baseline collection + P4 metric hardening
Week 4 (Sep 5–11)    P5 incremental validity + all figures/tables regenerated
Week 5 (Sep 12–18)   Rewrite paper against new evidence. ABSTRACT DEADLINE Sep 18.
Week 6 (Sep 19–25)   Polish, artifact freeze, Zenodo v2, camera-quality PDF. PAPER Sep 25.
```

---

## P0 — Hygiene (2 days, do first, blocks everything else)

These are cheap and they change what the numbers mean.

- [x] **Separate parse failure from wrong answer.** Add a `parse_status` field to every prediction record: `{ok, malformed_json, missing_field, refused, timeout, truncated}`. Report **Parse Rate** as a leaderboard column. Recompute every metric twice: (a) parse-failure-as-incorrect, (b) parse-failure-excluded. If the two differ by >3 pp for any model, the benchmark is partly measuring format compliance, and that must be said out loud.
  - *This alone may explain the Llama-3.1-8B 276/301.* Find out before writing another word about it.
- [x] **Un-gitignore `evaluation/baselines/`** or publish it as a Zenodo artifact with a manifest. "Reconstructs bit-exactly" is not verifiable if reviewers can't see the archive.
- [x] **Retire simulated tiers from the main leaderboard.** Move Expert/Strong/Competent/Heuristic/Random to a *separate* table labelled "harness validation baselines." Reviewers discount simulated rows, and mixing them into a ranked leaderboard reads as padding. Keep them — they justify the metric — just don't rank them alongside real models.
- [x] **Kill partial-coverage rows or complete them.** A row at n=59 next to a row at n=300 in the same table is a reviewer magnet. Either finish the run (see P1) or drop it to an appendix table with an explicit caveat.
- [ ] **Freeze a `v1.0` dataset tag** and pin it in the citation. Any later problem edits go to `v1.1`.
- [x] **Add a `results/schema.json`** for the prediction record format so third-party submissions are possible.

**Acceptance:** every number in the README is traceable to a committed JSON archive, and no metric silently absorbs a parse failure.

---

## P1 — Real evidence of failure: panel + human ceiling

### P1a — Expand the model panel to ≥ 12 real models (target 15)

The claim "capability drives calibration" is a claim about a *curve*. Five points, three of them simulated, is not a curve.

**⚠️ Read this before using the table below (2026-08-17).** The Groq row is largely
obsolete and the reason is itself a planning constraint: **four of the five panel
endpoints were withdrawn within fourteen weeks** — `qwen/qwen3-32b` and
`meta-llama/llama-4-scout-17b-16e-instruct` by 08-03, `llama-3.3-70b-versatile` and
`llama-3.1-8b-instant` by 08-17. Of the panel, only `openai/gpt-oss-120b` is still served.
Verify any endpoint with `client.models.list()` (the `openai` SDK against Groq's base URL;
raw `urllib` gets a meaningless Cloudflare 403) **before** planning a run around it, and
expect roughly a one-quarter-per-month attrition rate when scheduling multi-day
accumulation. Currently served and relevant: `openai/gpt-oss-120b`, `openai/gpt-oss-20b`,
`qwen/qwen3.6-27b`, `groq/compound`. The practical implication for this item is that
"expand to 12 models" now means *collect fast and archive everything*, because a row you
did not finish is a row you may never finish.

**Zero-cost sources:**

| Provider | Models to add | Cost | Notes |
|---|---|---|---|
| Groq free tier | ~~Llama-3.1-8B, Llama-3.3-70B, Llama-4-Scout, Qwen3-32B~~ (all withdrawn), GPT-OSS-20B/120B, Qwen3.6-27B, Llama-4-Maverick / Kimi-K2 *if still listed* | ₹0 | Daily token cap — checkpointing already exists, use it. Spread runs across days. Re-check the roster first. |
| OpenRouter `:free` | Nemotron, GLM-4.x, Hermes, DeepSeek-R1-distills, Mistral-small | ₹0 | Rate-limited, not token-capped. Good for overnight. |
| Google AI Studio free tier | Gemini 2.x Flash / Pro | ₹0 | **This is your one free frontier-class model. Do not skip it.** |
| Cerebras / SambaNova free tiers | Llama variants at high throughput | ₹0 | Worth 30 min to check current availability |

**Paid, only if funded (~$40–80 total for the whole panel):**
- One closed frontier model each from OpenAI / Anthropic (301 problems × ~2k output tokens ≈ $3–10 per model per full run).
- **Funding routes to try this week (all free to apply):** OpenAI Researcher Access Program, Anthropic External Researcher Access, Google Cloud research credits, AWS Cloud Credit for Research. Apply to all four on day 1 — turnaround is 1–3 weeks, which just fits.
- *Go/no-go:* if no credits land by **Sep 1**, ship with open-weights + Gemini only and retitle the contribution as an **open-weights** calibration study. That is a completely honest and still-publishable framing. Do not fake frontier coverage.

**Design requirement — deliberately span the capability axis.** Include models you *expect* to be bad (1–3B params) and models you expect to be good. A panel clustered at one capability level cannot test the capability hypothesis. Aim for ≥ 60 pp spread in Acc@CI across the panel.

- [ ] `models.yaml` manifest: model id, provider, param count (if known), release date, context, temperature, seed, run date, harness version.
- [ ] All models run on the **same** 301-problem test split, same prompt, same decoding config.
- [ ] Every model completes **full coverage** or is excluded. No partial rows in the main table.

### P1b — Human baseline (this is the single highest-value item in the plan)

Without this, "models are failing" is unfalsifiable. With it, you get a ceiling, a floor, and a headline.

- **Sample:** 60 problems, stratified — 12 per category, spanning D1–D5 (weight toward D3–D5 where failures live).
- **Participants:** 5–8 people with quantitative training (stats/CS/physics seniors, TA-level or above). Your SRMIST + CERN HSF + competitive-programming networks cover this. Target ~2 hours each.
- **Protocol:** same four-part output as models (point estimate, CI, confidence, framework). Calculator allowed, internet not. Timed. Collect think-aloud notes on 10 of the 60 per participant — these become qualitative evidence for P2.
- **Deliverables:**
  - Human Acc@CI, ECE, overconfidence, framework match — the **ceiling** row on the leaderboard.
  - **Inter-annotator agreement** on Adversarial Ambiguity items (Krippendorff's α). If humans disagree on the correct formalisation, your "closed-form ground truth" for that category needs a defensibility argument, not just a derivation.
  - Item-level human difficulty → validates or falsifies your D1–D5 labels. If human accuracy doesn't decline monotonically across D1→D5, your difficulty scale is a construct, not a measurement, and you must say so.
- [ ] Ethics: this is human-subjects data. Check SRMIST IRB/ethics requirements *now* — even if exempt, get the exemption in writing. Anonymous, consented, no PII stored.
- **Fallback if recruitment fails:** a single expert (you) + 2 others on 30 items, reported explicitly as a *pilot* human reference with n stated. Weaker, still far better than nothing.

**Acceptance:** the leaderboard has a human row with a CI, and you can write the sentence "the best model we evaluated is X pp below / above the human reference (95% CI [a, b])."

---

## P2 — "Failing on *what*": error taxonomy

This is the direct answer to the critic. Converts a number into a diagnosis.

### P2a — Build the taxonomy

**DONE 2026-08-17 for the rule-decidable half — and the codebook below was superseded by
one written after reading the errors.** See `evaluation/failure_codebook.py`, paper §7.2,
Table 18. What changed against the plan:

*Corpus (2026-08-16): `python evaluation/error_extract.py` writes
`evaluation/errors/errors.jsonl` (361 errors, 346 with readable responses) plus a seeded
sample. Corroborated unit mismatches are excluded from the sample — coding those would be
coding our own prompt, not the model.*

*Sampling: the note here previously said to sample proportionally with a floor, because
GPT-OSS-120B has only 9 codeable errors and Qwen3-32B 17. **Reversed.** Proportional
allocation answers "what does the panel's total error consist of"; the question P2c
actually asks — do failure modes shift with capability or merely shrink — is a
within-model composition question, and proportional allocation makes the sample
two-thirds Llama-3.1-8B. `error_extract.py` now draws **balanced** by default (cap 25 per
model, stratified by category within model) with `--proportional` for the old behaviour.
The clean end of the panel is still bounded by what exists: GPT-OSS contributes 9 however
the sample is drawn, and no per-model claim about the leader can escape that.*

*Codebook: the F1–F10 table below was written from the armchair and is kept only as a
record of what we expected to find. The codebook actually used was written after reading
the corpus, and its two largest categories are not on this list — answers right but in
another unit (24% of raw errors), and answers reporting a different quantity from the
same computation. It also splits detection: 5 codes decidable by a written-down rule, 7
needing a reasoning-chain read, with the uncoded remainder reported as a column. The
plan's F1 "prior neglect" collides with the paper's authored F1 "anchoring"; the new
codebook uses R/C/M/S prefixes to avoid that.*

Original plan text follows. Code a stratified sample of **300 incorrect predictions** (≈ 25 per model × 12 models, stratified by category and difficulty). Proposed initial codebook — refine after reading 50:

| Code | Failure mode |
|---|---|
| `F1` | **Prior neglect** — ignores or overwrites the stated prior |
| `F2` | **Base-rate neglect** — classic Bayesian error; likelihood dominates |
| `F3` | **Uncertainty collapse** — propagates a point estimate where the input was a distribution |
| `F4` | **Interval miscalibration — too narrow** (overconfident width) |
| `F5` | **Interval miscalibration — too wide** (uninformative hedge; must be penalised, see P4) |
| `F6` | **Framework misidentification** — right arithmetic, wrong named method |
| `F7` | **Ambiguity collapse** — picks one formalisation without acknowledging the alternative (Adversarial Ambiguity only) |
| `F8` | **Arithmetic / algebraic slip** — reasoning correct, execution wrong |
| `F9` | **Confidence-reasoning mismatch** — states low confidence, gives no interval widening (or vice versa) |
| `F10` | **Format / instruction-following failure** |

### P2b — Code it

- [x] **Rule-coded pass (not in the original plan, and it should have been).** Five of the twelve codes are decidable from the record without reading anything: R2 ambiguous-target, R3 wrong-summary, R4 unit-mismatch, M2 false-precision, S1 off-schema. `failure_codebook.py` decides those, unit-tested one rule at a time (17 tests, mostly negative cases), and returns *nothing* for the other seven rather than guessing. This is what makes the κ pass affordable: it removes 46 of 275 errors from the human queue and, more importantly, fixes the denominator before any judgment enters.
- [ ] **Human coding:** you code 100 items. A second coder codes an overlapping 50. Report **Cohen's κ**. Target κ > 0.7; if below, the codebook is ambiguous — revise and recode. *Sample ready: `evaluation/errors/sample_86.json`, balanced 25/25/25 + GPT-OSS 6 + Qwen 5, seeded and redrawable from the committed archives. The 86 excludes corroborated unit mismatches.*
- [ ] **LLM-judge scale-up:** use a strong model as a judge on the remaining 200, *validated against* the human-coded 100. Report judge–human agreement. Never report LLM-judge numbers without that validation figure — reviewers will (correctly) reject it otherwise. *Caveat that has appeared since: only `gpt-oss-120b` is still served on Groq (see P1a), so the judge is either that endpoint — judging its own errors among others — or a paid one.*
- [x] **Output:** a `model × failure-mode` matrix — for the mechanical codes, with `_uncoded` reported as a column. Per-difficulty breakdown still open; it needs the judgment codes to be worth splitting.

### P2c — Make it a finding, not an appendix

The interesting result to look for: **do failure modes shift with capability, or just shrink?**
- If weak and strong models fail the *same way* at different rates → miscalibration is a capability deficit (supports your current headline).
- If strong models fail *differently* (e.g. F8 arithmetic slips give way to F7 ambiguity collapse) → there is a qualitative transition, which is a much more interesting paper.

Either result is publishable. **Do not decide which one you want before you look.**

*Partial answer as of 2026-08-17, from the mechanical codes only, and it points at "shift"
rather than "shrink" — but it cannot yet carry that claim.* Of the errors surviving the
unit correction, a reporting or confidence code names 4 of 9 for GPT-OSS-120B and 6 of 17
for Qwen3-32B, against 3 of 39 for Llama-3.3-70B; false precision is 13/19 the 8B's. So at
the clean end of the panel a large share of what is left is *not arithmetic*. Two reasons
this is not yet the finding: 83% of surviving errors carry no mechanical code at all, so
the comparison is over the coded minority, and GPT-OSS's nine errors cannot support a
composition claim at any significance. **The κ pass is what converts this into the sentence
in the acceptance criterion.** A second, cleaner signal is already reportable: 21% of
surviving errors are wrong point estimates whose own stated interval contains the truth
(31% Llama-3.3, 33% Scout, 6% Qwen) — that is in the paper as §7.2's second finding.

- [ ] **6–8 worked failure examples** in the appendix: full problem, full model output, annotated with what went wrong and what the correct chain was. This is the single most persuasive artifact for a skeptical reader. One page of a model confidently botching a base rate does more than three tables.

**Acceptance:** you can complete the sentence "models fail on MURU-BENCH primarily by ___, and this shifts to ___ as capability increases" with a number and a CI attached.

---

## P3 — Robustness: is the failure real, or your prompt's fault?

This is where the "is it an artifact" objection dies. Run all ablations on a fixed **100-problem robustness subset** to keep cost down; report full-split only for the primary config.

- [ ] **A1 — Repeated sampling.** k = 5 samples per problem at T = 0.7, on ≥ 4 models spanning the capability range. Gives per-model variance, self-consistency, and lets you report whether the leaderboard ordering is stable under resampling. **Without this you have no error bars on individual model scores.**
- [ ] **A2 — Temperature sweep.** T ∈ {0.0, 0.3, 0.7, 1.0}. Does overconfidence track temperature? (Prediction: greedy decoding inflates stated confidence. Test it.)
- [ ] **A3 — Prompt-format ablation.** ≥ 3 variants: (i) current, (ii) reworded but semantically identical, (iii) different output-schema ordering. If Acc@CI moves > 5 pp between (i) and (ii), the benchmark is measuring prompt sensitivity and you must report the spread, not a point estimate. **Arm (ii) is done (2026-08-17)** — the v1→v2 prompt-version replication, which doubles as the validation of the unit correction (paper §6.3, `evaluation/compare_prompt_versions.py`, infrastructure in `data/robustness_subset.json` + `run_eval.py --ids/--tag`). Result: on GPT-OSS-120B and Llama-3.3-70B, v2 is indistinguishable from the v1 *corrected* scoring (McNemar p = 1.000 both) and moves 4.0 / 8.2 pp against the v1 *raw* scoring; unit mismatches 5→0 and 7→0. **Llama-3.1-8B is the warning the acceptance criterion was written for:** 29 of 94 paired problems flip, which is prompt sensitivity swamping the effect at that capability. Arms (iii) and the remaining variants are unaffected by endpoint withdrawal only for GPT-OSS — everything else in the panel is now uncollectable, so **A3 as originally scoped (≥3 variants × the panel) is no longer possible**; scope it to the surviving endpoint plus any newly-added models.
- [ ] **A4 — Confidence elicitation method.** Verbalized probability (current) vs. multi-sample empirical frequency vs. token-logprob-derived (where the API exposes logprobs). Known result in the literature: these disagree substantially. Showing *which* your benchmark measures — and how it compares — is a genuine contribution.
- [ ] **A5 — CoT vs. direct answer.** Does chain-of-thought improve calibration or just accuracy? Directly relevant to your headline.
- [ ] **A6 — Contamination / memorisation probe.** Regenerate the test split with a **new seed** (same templates, new parameter draws). If accuracy is statistically indistinguishable → strong evidence against memorisation and a real selling point of parametric generation. If it *moves*, you've found either contamination or template-level difficulty variance — both must be reported.
- [ ] **A7 — Few-shot.** 0-shot (current) vs. 3-shot with worked examples. Establishes whether failures are competence or task-comprehension.

**Acceptance:** a "Robustness" section stating, with numbers, how much of the reported effect survives each perturbation. Reviewers forgive instability that is *measured and disclosed*; they do not forgive instability that is discovered by them.

---

## P4 — Metric hardening

Acc@CI as currently defined is gameable: a model that emits [−∞, ∞] scores 100% coverage. You need sharpness.

- [x] **Add a proper scoring rule.** Interval Score / Winkler score for the CI (penalises width *and* miss simultaneously) and/or CRPS. Report alongside coverage. This closes the widest methodological hole in the paper.
- [x] **Report interval width distribution** per model — mean, median, and normalised-by-ground-truth-width.
- [x] **Coverage–sharpness plot.** Coverage on y, mean normalised width on x, one point per model. Immediately shows who is hedging.
- [x] **Harden ECE.** Report equal-mass (adaptive) binning *and* equal-width; report binning sensitivity across {5, 10, 15, 20} bins; add **smECE** or a debiased estimator. State the small-n bias of ECE at n=301 explicitly.
- [x] **Add Brier score with Murphy decomposition** (reliability / resolution / uncertainty). Separates "badly calibrated" from "uninformative," which ECE alone conflates.
- [x] **Add discrimination:** AUROC of stated confidence vs. correctness. A model can be badly calibrated but well-ranked; that distinction matters for downstream use.
- [x] **Formalise Overconfidence** with a stated threshold and a sensitivity analysis over that threshold. "High-confidence wrong" needs a defined cut and evidence the finding isn't cut-dependent.

**Acceptance:** every headline metric has a stated failure mode and a companion metric that covers it.

---

## P5 — Incremental validity: why does MURU-BENCH need to exist?

The question every benchmark paper must answer.

- [ ] Run the same panel on a **200-problem subset of MATH and GSM8K**, using the same harness.
- [ ] Compute **rank correlation** between MURU Acc@CI and MATH/GSM8K accuracy across the panel.
  - High correlation (ρ > 0.9) → MURU is redundant on point accuracy, and your value must come entirely from the calibration axis. Say so, and lean the paper on ECE/Interval Score, not Acc@CI.
  - Divergence → **this is your headline.** "Models that rank identically on MATH separate by X pp on MURU."
- [ ] Regress MURU calibration metrics on MATH accuracy. **Residual variance is the quantitative measure of what MURU adds.** That single number answers the whole objection.
- [ ] Explicit related-work positioning table vs. GSM8K, MATH, MMLU, and the existing LLM-calibration literature (verbalized-confidence work, selective-prediction benchmarks). What does no existing benchmark measure that yours does?

**Acceptance:** one sentence with a number: "MURU-BENCH explains X% of variance in model calibration that is not predicted by MATH accuracy."

---

## P6 — Paper, artifact, and accreditation

### Paper rewrite (Week 5)

Restructure around the new evidence:
1. Intro — the gap: benchmarks score answers, not uncertainty.
2. Benchmark construction (mostly reusable as-is; strongest section).
3. **Human baseline** (new) — establishes the ceiling.
4. Model panel + leaderboard (expanded, human-anchored).
5. **Failure taxonomy** (new) — the "on what" answer.
6. **Robustness** (new) — the "not an artifact" answer.
7. Metric validity + incremental validity.
8. Limitations — written honestly and *first*, not as an afterthought.

**Write §8 before §1.** If the limitations section is embarrassing, the experiments aren't done.

### Accreditation / external validation

The critic's word "accreditation" is about third-party trust, not more of your own numbers:

- [ ] **Croissant metadata validated** against the official validator (already have `metadata/croissant.json` — confirm it passes).
- [ ] **HuggingFace Datasets** release with a full dataset card + viewer. This is the single biggest discoverability and legitimacy win available, and it's free.
- [ ] **Zenodo v2** release including raw model responses, human baseline data, and coded failure annotations. Cite the versioned DOI.
- [ ] **Public leaderboard** with a documented submission protocol + CI-validated result format, so others can add models. A benchmark with an external submission path is a benchmark; one without is a paper.
- [ ] **Solicit adversarial review before submission.** Post the dataset + a call for errata to r/MachineLearning, the ML Collective / Eleuther Discords, and relevant HSF contacts. Every errata issue filed and fixed is evidence of quality control you can cite. Give it 2 weeks — start by **Sep 1**.
- [ ] **Get 2 external reads** of the full draft by Sep 15. Prateek is one. Find a second with benchmark or calibration-paper experience — a cold email to an author you cite has a non-trivial hit rate.

### ⚠️ ICLR 2027 eligibility check — do this on day 1

ICLR 2027 requires **at least one author registered to review**, qualified by having ≥ 1 accepted publication at ICLR/NeurIPS/ICML/UAI/AISTATS/JMLR/TMLR/ACL/EMNLP/etc. Workshop papers and Tiny Papers **do not** count.

As a solo first-year author, you likely do not meet this. **Verify the exact wording on the ICLR 2027 Author Guidelines page yourself before committing to the venue.** Options if you don't qualify:
1. Add a qualifying co-author (mentor, collaborator) with a genuine intellectual contribution. Not a courtesy authorship — a real one.
2. **Target TMLR instead.** Rolling deadline (no September crunch), no reciprocal-review requirement, well-regarded, and its "claims supported by evidence" review criterion is an *excellent* fit for a benchmark paper that has done P1–P5 properly. **This is probably the better primary target.**
3. ACL Rolling Review / ARR → EMNLP or NAACL.
4. NeurIPS 2027 D&B (May 2027 deadline) — gives ~9 months, which is the comfortable option if quality matters more than speed.

**Recommendation, stated plainly:** the 5.5-week ICLR window is achievable only if you cut the human baseline or the ablations, and those are exactly the two things the criticism is about. **Do the full P0–P5 work and submit to TMLR in October**, or hold for NeurIPS D&B 2027. Rushing to ICLR with half the evidence reproduces the exact weakness you're trying to fix.

---

## 2. Priority ranking (if you can only do some of it)

1. **P0** — cheap, changes what existing numbers mean. Non-negotiable.
2. **P1b human baseline** — without it, "failing" is not a measurable claim.
3. **P2 failure taxonomy** — the literal answer to "failing on what."
4. **P3 A1 + A3 + A6** (resampling, prompt variants, reseed) — kills the artifact objection.
5. **P1a panel to ≥12 models** — needed for any correlation claim.
6. **P4 Interval Score + ECE hardening** — closes the metric hole.
7. **P5 incremental validity** — answers "why does this exist."
8. **P6 accreditation** — HuggingFace + Zenodo v2 + public leaderboard.

---

## 3. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Headline finding does not survive n=12** | **Medium-high** | High | See §6. Pre-register the analysis; report the null honestly. |
| Human recruitment fails | Medium | High | Pilot fallback (3 coders × 30 items), reported as pilot with stated n |
| No API credits land | Medium | Medium | Retitle as open-weights study; Gemini free tier covers frontier-adjacent |
| Free-tier caps stall panel runs | High | Low | Checkpointing exists; start now, spread over 3 weeks |
| ICLR eligibility blocks submission | **High** | Medium | TMLR as primary target (see P6) |
| Adversarial Ambiguity ground truth doesn't survive human agreement check | Medium | **High** | If α is low, reframe that category as measuring *ambiguity acknowledgement*, not correctness — still a valid contribution, but rewrite the section |
| Scope creep eats the writing weeks | High | High | Hard freeze on new experiments after Sep 4 |

---

## 4. Definition of done

- [ ] ≥ 12 real models, full coverage, bootstrap CIs on every cell
- [ ] Human baseline row with n, CI, and IAA on the ambiguity subset
- [ ] Failure taxonomy over ≥ 300 coded errors with reported κ and judge validation *(codebook written from the corpus and the 5 rule-decidable codes applied to all 361 errors, 2026-08-17; the κ and judge halves are what remain — see P2b)*
- [ ] ≥ 5 robustness ablations with quantified effect sizes *(1 of 5: A3 arm (ii), the prompt-version replication, 2026-08-17. Note the ceiling: with four of five endpoints withdrawn, further ablations can only run on `gpt-oss-120b` or on newly-added models)*
- [x] Proper scoring rule + hardened ECE + discrimination metric *(P4, 2026-08-16)*
- [ ] Incremental-validity result vs. MATH/GSM8K with a residual-variance number
- [x] Parse rate reported; metrics computed both ways *(P0, 2026-08-15; unit accounting added as a third, 2026-08-16)*
- [ ] Raw responses + human data + annotations publicly archived
- [ ] HuggingFace dataset live; Croissant validated; Zenodo v2 minted
- [ ] Public leaderboard with documented submission protocol
- [ ] Limitations section that a hostile reviewer would call complete
- [ ] Two external reads incorporated

---

## 5. Immediate next 72 hours

1. Check ICLR 2027 reciprocal-reviewing eligibility. Decide venue. **This changes the whole timeline.**
2. Apply to all four credit programs (OpenAI, Anthropic, Google, AWS).
3. Ship P0: `parse_status` field, dual-metric recomputation, un-gitignore results. Find out what happened to those 25 missing Llama-3.1-8B items.
4. Launch Groq + OpenRouter panel runs with checkpointing — these run in background for two weeks.
5. Draft the human-baseline protocol + consent form; check SRMIST ethics requirements; send the recruitment ask.
6. Read 50 wrong answers by hand and draft the failure codebook from what you actually see, not from this document's guesses.

---

## 6. ⚠️ The one thing to be honest with yourself about

Your current headline is: *miscalibration is dominated by capability, not an independent metacognitive deficit* — supported by Spearman ρ = −0.90 over ~5 points, most of them simulated.

**With 12+ real models spanning a wider capability range, that correlation will very likely weaken.** Real panels almost always show more scatter than small ones, and simulated tiers are constructed to be internally consistent in a way real models are not.

Plan for that now:

- Write the analysis so the finding is **"how much of calibration variance is explained by capability"** (an R² with a CI), not **"calibration is capability"** (a binary claim that a single outlier can break).
- **Pre-register** the panel, the metrics, and the analysis before the runs finish. Commit it, timestamp it. Then you cannot be accused of fitting the story to the data — and more importantly, you won't.
- A partial correlation with identified outliers is a *better* paper than a suspiciously clean ρ = −0.90 over five points. Reviewers trust messy real results over clean small-n ones.
- If the correlation collapses entirely, that is still a finding, and arguably a more interesting one: it would mean calibration is a separable axis and MURU-BENCH is measuring something MATH cannot. That result would make P5 your headline instead of P1.

The dataset is genuinely good work. Let the evidence layer say whatever it says.
