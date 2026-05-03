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

## Baseline Results (Test Set, n=301)

| Model Tier | Accuracy | ECE ↓ | Overconfidence ↓ |
|------------|----------|-------|-----------------|
| Random | 7.3% | 0.515 | 36.2% |
| Heuristic | 31.2% | 0.470 | 44.5% |
| Competent (GPT-3.5 tier) | 49.2% | 0.239 | 21.6% |
| Strong (GPT-4 tier) | 60.8% | 0.178 | 20.3% |
| Expert (frontier) | 77.1% | 0.183 | 9.6% |

## Citation

```bibtex
@inproceedings{kumar2026murubench,
  title={MURU-BENCH: A Benchmark for Mathematical Reasoning Under Uncertainty},
  author={Kumar, Swetank},
  booktitle={NeurIPS Datasets and Benchmarks Track},
  year={2026},
  url={https://github.com/swetank18/MURU}
}
```

## License

MIT License
