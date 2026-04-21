# MURU-BENCH: Mathematical Reasoning Under Uncertainty Benchmark

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-blue.svg)](https://neurips.cc)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Problems](https://img.shields.io/badge/Problems-3%2C000-orange.svg)](data/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)

**MURU-BENCH** is a benchmark of **3,000 problems** for evaluating mathematical reasoning under genuine uncertainty. Unlike existing math benchmarks (GSM8K, MATH, MMLU) which test deterministic answers, MURU-BENCH requires models to produce **calibrated confidence intervals** and identify the correct **probabilistic reasoning framework**.

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **3,000 problems** | Spanning 5 categories and 5 difficulty levels |
| **Calibrated ground truth** | Each problem has a point estimate + confidence interval |
| **6 evaluation metrics** | Accuracy, ECE, overconfidence, framework match, and more |
| **21 parametric templates** | Reproducible, numerically diverse problem generation |
| **Multi-model support** | API clients for OpenAI, Anthropic, and Google models |

## 📊 Categories

| Category | Count | Description |
|----------|-------|-------------|
| Bayesian Updating | 910 | Bayes' theorem with uncertain priors/likelihoods |
| Distribution Estimation | 660 | Population parameter inference from finite samples |
| Decision Under Uncertainty | 525 | Expected utility with uncertain states |
| Adversarial Ambiguity | 474 | Problems with multiple valid formalizations |
| Conditional Prob. Chains | 431 | Multi-step conditioning with uncertain intermediates |

## 🏗️ Project Structure

```
MURU/
├── data/
│   ├── train/          # 2,398 problems (80%)
│   ├── validation/     # 301 problems (10%)
│   ├── test/           # 301 problems (10%)
│   ├── by_category/    # Symlinks organized by category
│   └── by_difficulty/  # Symlinks organized by difficulty
├── evaluation/
│   ├── metrics.py      # 6 core evaluation metrics
│   ├── run_eval.py     # API-based model evaluation
│   ├── run_baselines.py # Simulated baseline generator
│   ├── analyze_results.py # Analysis & LaTeX table generation
│   └── baselines/      # Saved baseline results
├── scripts/
│   ├── generate_problems.py  # 21 parametric templates
│   ├── validate.py           # Schema + semantic validation
│   ├── split_data.py         # Stratified train/val/test split
│   ├── generate_figures.py   # Publication-quality figures
│   ├── stats.py              # Dataset statistics
│   └── sample.py             # Problem inspector
├── paper/
│   ├── main.tex              # NeurIPS paper
│   ├── neurips_2024.sty      # Style file
│   └── figures/              # Generated figures
└── problem_schema.json       # JSON schema for problems
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/swetank18/MURU.git
cd MURU
pip install -r requirements.txt
```

### Inspect Problems

```bash
# Sample 3 random problems
python scripts/sample.py

# Sample D5 adversarial ambiguity problems
python scripts/sample.py --category adversarial_ambiguity --difficulty 5

# View a specific problem
python scripts/sample.py --id MURU-0001
```

### Evaluate a Model

```bash
# Set your API key
export OPENAI_API_KEY=your-key-here

# Run evaluation
python evaluation/run_eval.py --model gpt-4o --save

# Analyze results
python evaluation/analyze_results.py
```

### Run Simulated Baselines

```bash
python evaluation/run_baselines.py --save
```

## 📈 Baseline Results

Results on the test set (n=301):

| Model Tier | Accuracy | ECE ↓ | Overconfidence ↓ | Framework Match |
|------------|----------|-------|-----------------|-----------------|
| Random | 7.3% | 0.515 | 36.2% | 33.9% |
| Heuristic | 31.2% | 0.470 | 44.5% | 47.2% |
| Competent (GPT-3.5 tier) | 49.2% | 0.239 | 21.6% | 67.1% |
| Strong (GPT-4 tier) | 60.8% | 0.178 | 20.3% | 83.7% |
| Expert (frontier tier) | 77.1% | 0.183 | 9.6% | 89.0% |

**Difficulty scaling** — all models show monotonic accuracy decay:

| Model | D1 | D2 | D3 | D4 | D5 |
|-------|----|----|----|----|----|
| Expert | 96% | 91% | 81% | 64% | 21% |
| Strong | 88% | 78% | 63% | 33% | 14% |
| Competent | 81% | 72% | 44% | 22% | 4% |

## 🔬 Regenerate the Dataset

```bash
# Generate 3,000 problems from all 21 templates
python scripts/generate_problems.py --all --n 3000 --seed 2026 --validate

# Split into train/val/test
python scripts/split_data.py --source data/train/ --seed 42

# Generate paper figures
python scripts/generate_figures.py
```

## 📄 Citation

```bibtex
@inproceedings{kumar2026murubench,
  title={MURU-BENCH: A Benchmark for Mathematical Reasoning Under Uncertainty},
  author={Kumar, Swetank},
  booktitle={NeurIPS Datasets and Benchmarks Track},
  year={2026},
  url={https://github.com/swetank18/MURU}
}
```

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
