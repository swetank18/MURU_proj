.PHONY: test validate baselines bootstrap reanalyze eval-openai eval-anthropic eval-google eval-groq paper submission clean

PYTHON ?= .venv/bin/python

# Some local environments (e.g. ROS) inject incompatible plugins into PYTHONPATH;
# strip it so pytest only sees the venv's site-packages.
test:
	PYTHONPATH= $(PYTHON) -m pytest tests/ -v

validate:
	$(PYTHON) scripts/validate.py data/train/
	$(PYTHON) scripts/validate.py data/validation/
	$(PYTHON) scripts/validate.py data/test/

baselines:
	$(PYTHON) evaluation/run_baselines.py --save

# Reproduce the bootstrap CIs and McNemar tests reported in Section 5.
bootstrap:
	$(PYTHON) evaluation/bootstrap_analysis.py

# Rebuild every published leaderboard number from the committed archives:
# parse-status accounting, bootstrap CIs, hardened calibration metrics, the
# five LaTeX table includes, the figures, and the archive manifest. Requires
# no API key and no network access.
reanalyze:
	$(PYTHON) evaluation/parse_status.py
	$(PYTHON) evaluation/error_extract.py
	$(PYTHON) evaluation/failure_codebook.py
	$(PYTHON) evaluation/aggregate_real_llm.py
	$(PYTHON) scripts/generate_figures.py
	$(PYTHON) scripts/make_manifest.py

# Real-LLM evaluations on the test split (n=301). Each target requires the
# corresponding API key in the environment. Outputs land in evaluation/baselines/.
eval-openai:
	$(PYTHON) evaluation/run_eval.py --model gpt-4o --save
	$(PYTHON) evaluation/run_eval.py --model gpt-4o-mini --save

eval-anthropic:
	$(PYTHON) evaluation/run_eval.py --model claude-3-5-sonnet-latest --save
	$(PYTHON) evaluation/run_eval.py --model claude-3-5-haiku-latest --save

eval-google:
	$(PYTHON) evaluation/run_eval.py --model gemini-1.5-pro --save
	$(PYTHON) evaluation/run_eval.py --model gemini-1.5-flash --save

# Free tier (Llama 3.1 / Mixtral / Gemma) via Groq — set GROQ_API_KEY.
eval-groq:
	$(PYTHON) evaluation/run_eval.py --model llama-3.1-70b --save
	$(PYTHON) evaluation/run_eval.py --model llama-3.1-8b --save
	$(PYTHON) evaluation/run_eval.py --model mixtral-8x7b --save

paper:
	cd paper && ../tectonic main.tex

submission:
	cd paper && ../tectonic main.tex
	rm -f submission/muru-bench-neurips2026.zip
	rm -rf submission/figures submission/tables
	mkdir -p submission
	cp paper/main.tex paper/main.pdf paper/neurips_2024.sty submission/
	cp -r paper/figures submission/
	cp -r paper/tables submission/
	cd submission && zip -r muru-bench-neurips2026.zip main.tex main.pdf neurips_2024.sty figures/ tables/

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.bbl paper/*.blg
	rm -rf .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
