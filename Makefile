PYTHON ?= python3

.PHONY: test paper-evaluate paper-stats paper-figures paper-verify

test:
	$(PYTHON) -m pytest -q

paper-evaluate:
	$(PYTHON) paper_evaluation.py --artifacts drive_artifacts --output paper_outputs

paper-stats:
	$(PYTHON) run_statistical_pipeline.py --artifacts drive_artifacts --output paper_outputs

paper-figures:
	$(PYTHON) generate_figures.py --output paper_outputs

paper-verify:
	$(PYTHON) verify_deliverables.py --output paper_outputs
