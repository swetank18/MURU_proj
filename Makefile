.PHONY: test validate baselines paper clean

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

paper:
	cd paper && ../tectonic main.tex

clean:
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.bbl paper/*.blg
	rm -rf .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
