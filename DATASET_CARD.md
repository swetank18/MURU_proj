---
license: mit
task_categories:
  - question-answering
  - text-generation
language:
  - en
tags:
  - mathematical-reasoning
  - uncertainty-quantification
  - calibration
  - benchmark
  - bayesian-inference
  - confidence-intervals
pretty_name: MURU-BENCH
size_categories:
  - 1K<n<10K
---

# MURU-BENCH: Mathematical Reasoning Under Uncertainty Benchmark

## Dataset Description

**MURU-BENCH** is a benchmark of **3,000 problems** for evaluating LLMs on mathematical reasoning under genuine uncertainty. Unlike GSM8K, MATH, or MMLU (which test deterministic answers), MURU-BENCH requires models to produce **calibrated confidence intervals** and identify the correct **probabilistic reasoning framework**.

### Key Features
- 🎯 **3,000 problems** across 5 categories and 5 difficulty levels
- 📊 **Calibrated ground truth** — each problem has a point estimate + confidence interval
- 📐 **6 evaluation metrics** — accuracy, ECE, overconfidence, framework match
- 🏗️ **21 parametric templates** — reproducible, numerically diverse
- 🔬 **Adversarial Ambiguity** — problems with multiple valid formalizations

## Categories

| Category | Count | Description |
|----------|-------|-------------|
| Bayesian Updating | 910 | Bayes' theorem with uncertain priors/likelihoods |
| Distribution Estimation | 660 | Population inference from finite samples |
| Decision Under Uncertainty | 525 | Expected utility with uncertain states |
| Adversarial Ambiguity | 474 | Multiple valid formalizations |
| Conditional Probability Chains | 431 | Multi-step conditional reasoning |

## Difficulty Levels

| Level | Description | Expert Accuracy |
|-------|-------------|-----------------|
| D1 | Single-formula application | 96% |
| D2 | Two-step reasoning | 91% |
| D3 | Multi-step with uncertainty | 81% |
| D4 | Compound uncertainty | 64% |
| D5 | Multi-step + adversarial | 21% |

## Data Format

Each problem is a JSON file with:
```json
{
  "id": "MURU-0001",
  "stem": "A medical test for a rare disease...",
  "category": "bayesian_updating",
  "difficulty": 2,
  "uncertainty_type": "epistemic_prior",
  "required_framework": "bayesian_inference",
  "ground_truth": {
    "answer": "The posterior probability is...",
    "point_estimate": 0.087,
    "confidence_interval": [0.046, 0.126],
    "ci_level": 0.95
  },
  "solution_steps": ["Step 1: ...", "Step 2: ..."],
  "common_failure_modes": ["Base rate neglect", "..."]
}
```

## Splits

| Split | Count | Purpose |
|-------|-------|---------|
| Train | 2,398 | Fine-tuning |
| Validation | 301 | Hyperparameter selection |
| Test | 301 | Evaluation (reported results) |

Stratified by (category, difficulty) for balanced representation.

## Evaluation

Models are scored on 6 metrics:
- **Accuracy@CI**: Point estimate within ground-truth confidence interval
- **ECE**: Expected Calibration Error (lower is better)
- **Overconfidence Rate**: High-confidence wrong answers (lower is better)
- **Framework Match**: Correct reasoning framework identified
- **Category Breakdown**: Per-category accuracy
- **Difficulty Scaling**: Per-difficulty accuracy

## Language-Model Results (Test Set, n=301)

Five hosted open-weights models. All columns are **unit-aware**: predictions stated in a
different admissible unit are read in the ground truth's unit before scoring, under the
corroboration rule in `evaluation/unit_accounting.py`. See the README for the same table
under the original scoring.

| Model | Accuracy@CI | ECE ↓ | Overconfidence ↓ |
|---|---:|---:|---:|
| GPT-OSS-120B | 97.0% | 0.067 | 3.0% |
| Qwen3-32B (245/301) | 93.1% | 0.117 | 5.7% |
| Llama-4-Scout-17B | 88.3% | 0.052 | 10.0% |
| Llama-3.3-70B | 87.0% | 0.051 | 12.0% |
| Llama-3.1-8B | 43.9% | 0.468 | 50.2% |

Accuracy and calibration-in-level are **separable** on this panel: Spearman ρ between
Accuracy@CI and ECE is −0.10 (exact p = 0.95, n = 5). Four of the five endpoints have
since been withdrawn by the provider; every raw API response is committed, so the numbers
rebuild from the archives without a live endpoint.

## Harness-Validation Baselines (simulated — **not** a ranking of real models)

These five rows are **synthetic capability tiers** produced by `evaluation/run_baselines.py`,
used to establish that the 301-problem split can discriminate tier-scale differences. The
tier names describe the configured profile, not a measurement of any named model.

| Simulated Tier | Accuracy | ECE ↓ | Overconfidence ↓ |
|------------|----------|-------|-----------------|
| Random | 7.3% | 0.515 | 36.2% |
| Heuristic | 31.2% | 0.470 | 44.5% |
| Competent | 49.2% | 0.239 | 21.6% |
| Strong | 60.8% | 0.178 | 20.3% |
| Expert | 77.1% | 0.183 | 9.6% |

## Known Limitations

Three item-construction defect classes are known. They are **fixed at the generator and shipped as a `v1.1` errata set**, not patched into `v1.0`: `v1.0` is the corpus the model panel answered, the committed archives hold responses to those exact stems, and four of the five endpoints have since been withdrawn — so an edited stem would leave an archived answer attached to a question nobody asked. `v1.0` stays reproducible; `errata/v1.1/` carries the 280 repaired items.

They were found by reading model errors (paper §7.3), which only surfaces a defect a model happened to trip over. `scripts/audit_item_defects.py` checks the corpus directly and is the authority on the count.

| Defect | Affected | Effect |
|---|---|---|
| **D1** Physically implausible stem values — mean diastolic BP of 213.1 / 482.3 / 280.8 mmHg, fuel consumption up to 424 L/100km, per-hectare yields up to 471 tonnes | 64 items (`MURU-0422`, `MURU-1258`, `MURU-1909`, …) | A model that applies domain knowledge and rejects the value is scored **wrong**; models that copy it through are scored right |
| **D2** Base-rate template is internally inconsistent — stem accuracy ≠ the colleague's quoted figure, and ground truth silently uses the latter as sensitivity | 170 of 185 base-rate items; 131 unsolvable as written | Read as overall accuracy the stem implies a sensitivity above 1, so the item has no consistent answer |
| **D3** Ground-truth intervals narrower than the precision the prompt invites (as tight as 0.001) | 37 findings across Simpson's-paradox, hierarchical-Bayes and saturated redundancy items | Arithmetically correct answers reported at the invited precision fall outside |

**280 of 3,000 problems** are affected — 219 train, 32 validation, **29 of 301 test**. The validators missed all of it because they check internal consistency and schema conformance, not plausibility; `make audit` checks it now. Two harness-side parsing defects found in the same pass are fixed (paper §7.3).

**Does the published result depend on them?** No. Re-scoring every archive on the clean 272 test items (`evaluation/defect_leaveout.py`) moves no model by more than 1.0 pp, leaves the leaderboard ordering unchanged, and leaves the headline null intact (accuracy/ECE Spearman −0.10 → −0.20). Every model scores at or above its own average on the 29 affected items, so the defects were mildly inflating scores rather than depressing them.

Further limitations of the evaluation, rather than the data: the language-model panel is five models, four of whose endpoints have since been withdrawn by the provider; the panel was collected under a prompt that did not state the unit convention (corrected post hoc, and validated by a paired re-run); and one row is at partial coverage (Qwen3-32B, 245/301).

## Citation

```bibtex
@inproceedings{kumar2026murubench,
  title={MURU-BENCH: A Benchmark for Mathematical Reasoning Under Uncertainty},
  author={Kumar, Swetank},
  booktitle={NeurIPS Datasets and Benchmarks Track},
  year={2026},
  url={https://github.com/swetank18/MURU_proj}
}
```

## License

MIT License
