# Contributing to MURU-BENCH

Thank you for your interest in contributing to MURU-BENCH! This document explains how to add new problems to the benchmark.

## Problem Quality Bar

A MURU-BENCH problem is accepted if and only if **ALL** of these are true:

1. **Unambiguous correctness** — A professional mathematician would agree the answer is correct given the stated assumptions
2. **Measurable failure** — A confident but wrong model would get it wrong in a measurable way
3. **Mathematical uncertainty** — The uncertainty is mathematical, not just linguistic vagueness
4. **Active reasoning required** — Cannot be solved by retrieval alone
5. **Non-trivial CI** — The ground truth confidence interval is not [0, 0] or [1, 1]

### Five more, added 2026-08-17 because real items failed them

Criterion 1 is about correctness *given the stated assumptions*, and three defective items passed it while having assumptions that were impossible, inconsistent, or unstated. These checks are what would have caught them (paper §7.3):

6. **Physically possible quantities** — every number in the stem lies in the real range of the quantity it names. Three shipped items state mean diastolic blood pressures of 213–482 mmHg. A model that knows physiology is scored **wrong** for objecting, which inverts what the benchmark is for.
7. **Internally consistent statistics** — if a stem gives more than one of accuracy / sensitivity / specificity / base rate, they must be mutually satisfiable, and the ground truth must use the figures the stem actually states. The base-rate-fallacy template quotes two different accuracies and takes ground truth from the one that is not in the stem; solved as written it implies a sensitivity above 1.
8. **Interval wider than the invited precision** — if the answer is naturally reported to three decimals, a ground-truth interval 0.001 wide fails correct arithmetic on rounding. Either widen by a rounding allowance or state the required precision in the prompt.
9. **One question, one number** — the schema accepts a single point estimate, so a stem must not ask three things and leave the scorer to pick. Items asking "compute the expected value of each option" are answered with the *other* option's value by four of five models, which is a defect of the item.
10. **State the unit** — if a quantity is denominated in \$K or in percent, say so. A quarter of the panel's recorded errors were right answers in another admissible unit.

If you are unsure whether an item clears 6–10, submit it anyway and say so in the PR: an item that fails one of these is more useful as a documented example than as a silent scoring error.

## Problem Writing Workflow

### Step 1: Choose a Scenario
Pick a real-world scenario where uncertainty is inherent:
- Medical testing
- Quality control
- Financial forecasting
- Scientific measurement
- Environmental monitoring

### Step 2: Embed Mathematical Structure
Choose from:
- Bayes' theorem
- Conditional probability
- Law of total probability
- Likelihood ratios
- Bayesian credible intervals

### Step 3: Introduce Uncertainty
Add genuine uncertainty through:
- A parameter range (not a single value)
- A missing or unknown prior
- A biased sample
- An unreliable information source

### Step 4: Derive Ground Truth
**Solve the problem yourself completely**, showing all steps. If you cannot derive the answer, the problem is not ready.

### Step 5: Identify Failure Modes
Write out explicitly what an overconfident model would do wrong. Common patterns:
- Ignoring parameter uncertainty (giving a point answer when a range is needed)
- Using the wrong framework
- Failing to propagate uncertainty through intermediate steps
- Treating a biased sample as representative

### Step 6: Validate and Commit

```bash
# Validate your problem
python scripts/validate.py data/train/MURU-XXXX.json

# Check it displays correctly
python scripts/sample.py --id MURU-XXXX
```

## JSON Schema

Every problem must conform to `problem_schema.json`. Here's a template:

```json
{
  "id": "MURU-XXXX",
  "category": "bayesian_updating",
  "difficulty": 3,
  "stem": "Your problem statement here (min 50 chars)...",
  "uncertainty_type": "parameter_uncertainty",
  "required_framework": "bayesian_inference",
  "ground_truth": {
    "answer": "Human-readable answer with range (min 10 chars)",
    "point_estimate": 0.81,
    "confidence_interval": [0.71, 0.89],
    "ci_level": 0.90
  },
  "solution_steps": [
    "Step 1: Identify the prior probabilities...",
    "Step 2: Calculate the likelihood given the data..."
  ],
  "common_failure_modes": [
    "Ignores parameter uncertainty and gives a single point answer"
  ],
  "metadata": {
    "author": "your_id",
    "reviewed": false,
    "source_inspiration": "original"
  }
}
```

### Field Reference

| Field | Type | Allowed Values |
|-------|------|---------------|
| `category` | string | `bayesian_updating`, `conditional_probability_chains`, `distribution_estimation`, `decision_under_uncertainty`, `adversarial_ambiguity` |
| `difficulty` | integer | 1–5 |
| `uncertainty_type` | string | `parameter_uncertainty`, `model_uncertainty`, `data_uncertainty`, `structural_uncertainty`, `epistemic_uncertainty` |
| `required_framework` | string | `bayesian_inference`, `frequentist_inference`, `decision_theory`, `information_theory`, `monte_carlo` |
| `source_inspiration` | string | `original`, `textbook`, `paper`, `real_world` |

## Naming Convention

- File name must match the `id` field: `MURU-0042.json` contains `"id": "MURU-0042"`
- IDs are sequential: check the latest ID before creating new problems

## Pull Request Process

1. Create your problems in `data/train/`
2. Run `python scripts/validate.py` — all must pass
3. Run `python scripts/stats.py` — verify counts
4. Submit a PR with a brief description of the problems added
