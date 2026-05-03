# Datasheet for MURU-BENCH

*Following the framework of [Gebru et al. (2021), "Datasheets for Datasets"](https://arxiv.org/abs/1803.09010).*

## Motivation

### For what purpose was the dataset created?
MURU-BENCH was created to evaluate large language models' ability to perform mathematical reasoning under genuine uncertainty — a capability untested by existing benchmarks (GSM8K, MATH, MMLU) which exclusively use deterministic answers. The benchmark fills the "calibrated uncertainty gap": no prior benchmark requires models to produce calibrated confidence intervals as part of their answer.

### Who created the dataset and on behalf of which entity?
Swetank Kumar, Department of Computer Science, SRM Institute of Science and Technology (SRMIST), Chennai, India. This is an independent research project.

### Who funded the creation of the dataset?
This is unfunded independent research. No external grants or sponsorships were involved.

## Composition

### What do the instances that comprise the dataset represent?
Each instance is a mathematical problem involving genuine uncertainty. Problems require identifying the correct probabilistic framework, computing point estimates, and producing calibrated confidence intervals.

### How many instances are there in total?
3,000 problems, split into:
- Train: 2,398 (80%)
- Validation: 301 (10%)
- Test: 301 (10%)

### Does the dataset contain all possible instances or is it a sample?
It is a curated set generated from 21 parametric templates. Each template can generate unlimited additional instances with different numerical parameters.

### What data does each instance consist of?
Each instance is a JSON document with:
- `id`: Unique identifier (e.g., MURU-0001)
- `stem`: Natural-language problem statement
- `category`: One of 5 categories
- `difficulty`: Integer 1-5
- `uncertainty_type`: Type of mathematical uncertainty
- `required_framework`: Correct reasoning framework
- `ground_truth`: Object with `point_estimate`, `confidence_interval`, `ci_level`
- `solution_steps`: Ordered list of reasoning steps
- `common_failure_modes`: Typical errors
- `metadata`: Author, template, timestamp

### Is there a label or target associated with each instance?
Yes. Each problem has a ground-truth `point_estimate` and `confidence_interval`. A model's answer is correct if its point estimate falls within the ground-truth CI.

### Is any information missing from individual instances?
No. All fields are required and validated by an automated schema checker.

### Are relationships between individual instances made explicit?
Problems from the same template share mathematical structure but differ in numerical parameters and domain context. The `metadata.source_template` field identifies the template.

### Are there recommended data splits?
Yes. We provide pre-computed stratified splits (train/validation/test) balanced across all 25 cells of the category × difficulty matrix.

### Are there any errors, sources of noise, or redundancies?
- 28 problems from the Value-of-Information template had a non-monotonicity bug in CI computation, which was fixed (documented in paper Section A.4)
- All 3,000 problems pass automated schema and semantic validation
- No known duplicate stems exist

### Does the dataset contain data that might be considered confidential?
No. All problems are synthetically generated mathematical problems.

### Does the dataset contain data that, if viewed directly, might be offensive?
No. Problems use neutral domain contexts (medical testing, manufacturing, finance) without any sensitive or offensive content.

## Collection Process

### How was the data associated with each instance acquired?
All problems are synthetically generated using 21 parametric templates implemented in `scripts/generate_problems.py`. No data was collected from human subjects or scraped from the web.

### What mechanisms or procedures were used to collect the data?
Parametric generation with:
1. Random parameter sampling within template-defined ranges
2. Domain context selection from pre-defined pools
3. Closed-form solution computation
4. Automated validation (schema + semantic consistency)

### If the dataset is a sample from a larger set, what was the sampling strategy?
Not applicable — the dataset is a complete generation run, not a sample.

### Who was involved in the data collection process?
Swetank Kumar (sole author) designed the templates and generation pipeline.

### Over what timeframe was the data collected?
April 2026.

### Were any ethical review processes conducted?
Not applicable — the dataset contains only synthetic mathematical problems with no human subjects data.

## Preprocessing/Cleaning/Labeling

### Was any preprocessing/cleaning/labeling of the data done?
Yes:
1. Schema validation: All required fields present, correct types
2. Semantic validation: Point estimate within CI, non-degenerate CI width
3. Bug fix: 28 problems regenerated after VOI template bug (documented)

### Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data?
The generation process is deterministic given a seed, so the raw data can be regenerated: `python scripts/generate_problems.py --all --n 3000 --seed 2026`

## Uses

### Has the dataset been used for any tasks already?
Yes — simulated baseline evaluation with 5 capability tiers, described in the accompanying paper.

### Is there a repository that links to any or all papers or systems that use the dataset?
https://github.com/swetank18/MURU

### What (other) tasks could the dataset be used for?
- Fine-tuning LLMs for calibrated mathematical reasoning
- Studying overconfidence and calibration in language models
- Evaluating chain-of-thought prompting strategies for uncertainty
- Studying adversarial robustness in mathematical reasoning

### Is there anything about the composition of the dataset or the way it was collected that might impact future uses?
- Template-based generation may introduce exploitable structural patterns
- English-only limits cross-lingual evaluation
- Category coverage excludes stochastic processes and measure theory

## Distribution

### Will the dataset be distributed to third parties outside of the entity on behalf of which the dataset was created?
Yes — the dataset is publicly available under the MIT License.

### How will the dataset be distributed?
Via GitHub: https://github.com/swetank18/MURU

### When was the dataset first released?
April 2026 (initial release).

### Will the dataset be updated?
Future versions may add new categories, more D5 problems, and additional templates.

### Is there an applicable EULA or terms of use?
MIT License — free for any use with attribution.

## Maintenance

### Who is supporting/hosting/maintaining the dataset?
Swetank Kumar (swetankkumar391@gmail.com)

### How can the owner/curator/manager of the dataset be contacted?
Email: swetankkumar391@gmail.com
GitHub: https://github.com/swetank18

### Will the dataset be updated? How often? By whom?
Updates planned for additional categories and difficulty levels. No fixed schedule.

### If the dataset relates to people, are there applicable limits on the retention of the data?
Not applicable — no personal data.

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?
Yes — see CONTRIBUTING.md in the repository. New templates can be added to `scripts/generate_problems.py`.
