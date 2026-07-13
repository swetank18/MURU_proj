# MURU-BENCH — Camera-Ready LaTeX Submission

Self-contained LaTeX source for the paper:

> **MURU-BENCH: A Benchmark for Mathematical Reasoning Under Uncertainty**
> Swetank Kumar, SRM Institute of Science and Technology (SRMIST), Chennai.
> Prepared for the NeurIPS 2026 Datasets & Benchmarks Track.

This folder compiles on its own — it does **not** depend on the rest of the
repository. Everything the document needs is here.

## Contents

| Path | Description |
|------|-------------|
| `main.tex` | Full paper source. Uses an inline `thebibliography` (no external `.bib`). |
| `main.pdf` | Pre-built PDF (27 pages), for reference. |
| `neurips_2024.sty` | NeurIPS style file. |
| `tables/` | Auto-generated result tables, `\input{}` by `main.tex`. Regenerated bit-for-bit by `evaluation/aggregate_real_llm.py` in the main repo. |
| `figures/` | All figures (PNG). |
| `muru-bench-neurips2026.zip` | Zipped copy of this folder for one-click upload to OpenReview. |

## Build

Any modern TeX toolchain works. The document is single-pass (inline
bibliography), so no separate `bibtex` run is required.

```bash
# Option A — tectonic (self-contained, no system TeX install needed)
tectonic main.tex

# Option B — a standard TeX Live install
pdflatex main.tex && pdflatex main.tex   # twice, to settle cross-references
```

The build is expected to finish with **no undefined references** and produce a
27-page PDF. (A handful of harmless `Overfull/Underfull \hbox` and
UTF-8-in-`.sty` warnings are emitted by the NeurIPS style file and can be
ignored.)

## Reproducing the numbers in the tables

The `tables/*.tex` files are not hand-edited — they are generated from the raw
model-response archives in `evaluation/baselines/` of the main repository:

```bash
python evaluation/aggregate_real_llm.py
```

This rewrites `paper/tables/real_llm_{main,difficulty,category}.tex`
deterministically from the archives, so every row in the paper reconstructs
from committed data.
