# MURU-BENCH

**Mathematical Reasoning Under Uncertainty Benchmark** — a dataset of 3,000 problems for evaluating whether language models can reason correctly when the answer is genuinely uncertain.

Existing math benchmarks (GSM8K, MATH, MMLU) test deterministic answers: a single correct number. MURU-BENCH instead requires models to (1) identify the right probabilistic framework, (2) execute multi-step reasoning under parameter uncertainty, and (3) return a calibrated point estimate together with a justified confidence interval.

---

## At a glance

| | |
|---|---|
| Problems | 3,000 |
| Categories | 5 (Bayesian Updating, Conditional Chains, Distribution Estimation, Decision Under Uncertainty, Adversarial Ambiguity) |
| Difficulty levels | 5 (D1 easiest, D5 hardest) |
| Generation templates | 21 parametric templates with seeded RNG |
| Splits | 2,398 train / 301 validation / 301 test (stratified by category × difficulty) |
| Evaluation metrics | Accuracy@CI, ECE, Overconfidence, Framework Match, plus per-category and per-difficulty breakdowns |
| Model APIs supported | OpenAI, Anthropic, Google |
| Tests | 26 pytest cases, run on Python 3.10 / 3.11 / 3.12 in CI |

---

## Categories

| Category | Count | What it tests |
|---|---:|---|
| Bayesian Updating | 910 | Bayes' theorem with uncertain priors or likelihoods |
| Distribution Estimation | 660 | Population parameter inference from finite samples |
| Decision Under Uncertainty | 525 | Expected utility when the state is uncertain |
| Adversarial Ambiguity | 474 | Problems with multiple valid formalizations |
| Conditional Probability Chains | 431 | Multi-step conditioning with uncertain intermediates |

---

## Repository layout

```
MURU/
  data/
    train/                # 2,398 problems
    validation/           # 301 problems
    test/                 # 301 problems
    by_category/          # Symlinks organized by category
    by_difficulty/        # Symlinks organized by difficulty
  evaluation/
    metrics.py            # Six evaluation metrics
    run_eval.py           # API-driven model evaluation (OpenAI/Anthropic/Google)
    run_baselines.py      # Simulated baseline tiers
    analyze_results.py    # Generates LaTeX tables and analysis report
    baselines/            # Saved baseline outputs (gitignored)
  scripts/
    generate_problems.py  # Parametric problem generator
    validate.py           # Schema + semantic validation
    split_data.py         # Stratified train/val/test split
    generate_figures.py   # Paper figures
    stats.py              # Dataset statistics
    sample.py             # Problem inspector
  tests/                  # pytest suite
  paper/
    main.tex              # NeurIPS 2026 paper source
    main.pdf              # Compiled PDF
    neurips_2024.sty      # Style file
    figures/              # Generated figures
  .github/workflows/      # CI configuration
  problem_schema.json     # JSON Schema for problems
  Makefile                # Convenience targets
  pytest.ini              # pytest configuration
  requirements.txt        # Python dependencies
```

---

## Installation

```bash
git clone https://github.com/swetank18/MURU.git
cd MURU
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick start

### Inspect problems

```bash
python scripts/sample.py                                                  # 3 random
python scripts/sample.py --category adversarial_ambiguity --difficulty 5  # filtered
python scripts/sample.py --id MURU-0001                                   # specific
```

### Validate the dataset

```bash
make validate
```

### Run simulated baselines (no API key needed)

```bash
python evaluation/run_baselines.py --save
python evaluation/analyze_results.py
```

### Evaluate a real model

```bash
export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY / GOOGLE_API_KEY
python evaluation/run_eval.py --model gpt-4o --save
python evaluation/analyze_results.py
```

### Run the test suite

```bash
make test
```

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs the same suite against Python 3.10, 3.11, and 3.12 on every push and pull request.

---

## Baseline results

Results on the test split (n = 301), arrows indicate desired direction.

| Model tier | Accuracy@CI | ECE (lower is better) | Overconfidence (lower) | Framework Match |
|---|---:|---:|---:|---:|
| Random | 7.3% | 0.515 | 36.2% | 33.9% |
| Heuristic | 31.2% | 0.470 | 44.5% | 47.2% |
| Competent (GPT-3.5 tier) | 49.2% | 0.239 | 21.6% | 67.1% |
| Strong (GPT-4 tier) | 60.8% | 0.178 | 20.3% | 83.7% |
| Expert (frontier tier) | 77.1% | 0.183 | 9.6% | 89.0% |

### Difficulty scaling

All tiers show monotonic accuracy decay from D1 to D5; D5 separates capability levels most cleanly.

| Model | D1 | D2 | D3 | D4 | D5 |
|---|---:|---:|---:|---:|---:|
| Expert | 96% | 91% | 81% | 64% | 21% |
| Strong | 88% | 78% | 63% | 33% | 14% |
| Competent | 81% | 72% | 44% | 22% | 4% |

---

## Regenerate the dataset

```bash
python scripts/generate_problems.py --all --n 3000 --seed 2026 --validate
python scripts/split_data.py --source data/train/ --seed 42
python scripts/generate_figures.py
```

The generator is deterministic given the seed.

---

## Make targets

| Target | Description |
|---|---|
| `make test` | Run the full pytest suite |
| `make validate` | Schema-validate all 3,000 problem JSONs |
| `make baselines` | Run all five simulated baselines and save outputs |
| `make paper` | Compile the LaTeX paper with tectonic |
| `make clean` | Remove build artifacts and pytest cache |

---

## Citation

```bibtex
@inproceedings{kumar2026murubench,
  title     = {MURU-BENCH: A Benchmark for Mathematical Reasoning Under Uncertainty},
  author    = {Kumar, Swetank},
  booktitle = {NeurIPS Datasets and Benchmarks Track},
  year      = {2026},
  url       = {https://github.com/swetank18/MURU}
}
```

---

## License

MIT. See [LICENSE](LICENSE).
