# MURU-BENCH — Completion Plan v1.0

## Progress log

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
taxonomy, P3 A1/A3/A6 ablations, P1a panel expansion, P5, P6. Nothing below has been
started.

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

**Zero-cost sources:**

| Provider | Models to add | Cost | Notes |
|---|---|---|---|
| Groq free tier | Llama-3.1-8B, Llama-3.3-70B, Llama-4-Scout, Llama-4-Maverick, GPT-OSS-20B/120B, Qwen3-32B, Kimi-K2 | ₹0 | Daily token cap — checkpointing already exists, use it. Spread runs across days. |
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

Code a stratified sample of **300 incorrect predictions** (≈ 25 per model × 12 models, stratified by category and difficulty). Proposed initial codebook — refine after reading 50:

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

- [ ] **Human coding:** you code 100 items. A second coder codes an overlapping 50. Report **Cohen's κ**. Target κ > 0.7; if below, the codebook is ambiguous — revise and recode.
- [ ] **LLM-judge scale-up:** use a strong model as a judge on the remaining 200, *validated against* the human-coded 100. Report judge–human agreement. Never report LLM-judge numbers without that validation figure — reviewers will (correctly) reject it otherwise.
- [ ] **Output:** a `model × failure-mode` matrix, plus per-difficulty breakdown.

### P2c — Make it a finding, not an appendix

The interesting result to look for: **do failure modes shift with capability, or just shrink?**
- If weak and strong models fail the *same way* at different rates → miscalibration is a capability deficit (supports your current headline).
- If strong models fail *differently* (e.g. F8 arithmetic slips give way to F7 ambiguity collapse) → there is a qualitative transition, which is a much more interesting paper.

Either result is publishable. **Do not decide which one you want before you look.**

- [ ] **6–8 worked failure examples** in the appendix: full problem, full model output, annotated with what went wrong and what the correct chain was. This is the single most persuasive artifact for a skeptical reader. One page of a model confidently botching a base rate does more than three tables.

**Acceptance:** you can complete the sentence "models fail on MURU-BENCH primarily by ___, and this shifts to ___ as capability increases" with a number and a CI attached.

---

## P3 — Robustness: is the failure real, or your prompt's fault?

This is where the "is it an artifact" objection dies. Run all ablations on a fixed **100-problem robustness subset** to keep cost down; report full-split only for the primary config.

- [ ] **A1 — Repeated sampling.** k = 5 samples per problem at T = 0.7, on ≥ 4 models spanning the capability range. Gives per-model variance, self-consistency, and lets you report whether the leaderboard ordering is stable under resampling. **Without this you have no error bars on individual model scores.**
- [ ] **A2 — Temperature sweep.** T ∈ {0.0, 0.3, 0.7, 1.0}. Does overconfidence track temperature? (Prediction: greedy decoding inflates stated confidence. Test it.)
- [ ] **A3 — Prompt-format ablation.** ≥ 3 variants: (i) current, (ii) reworded but semantically identical, (iii) different output-schema ordering. If Acc@CI moves > 5 pp between (i) and (ii), the benchmark is measuring prompt sensitivity and you must report the spread, not a point estimate.
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
- [ ] **Formalise Overconfidence** with a stated threshold and a sensitivity analysis over that threshold. "High-confidence wrong" needs a defined cut and evidence the finding isn't cut-dependent.

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
- [ ] Failure taxonomy over ≥ 300 coded errors with reported κ and judge validation
- [ ] ≥ 5 robustness ablations with quantified effect sizes
- [ ] Proper scoring rule + hardened ECE + discrimination metric
- [ ] Incremental-validity result vs. MATH/GSM8K with a residual-variance number
- [ ] Parse rate reported; metrics computed both ways
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
