# strong-crane — pipeline targets (run from the repository root; PY points at the environment with requirements.txt)
PY ?= python
# the final organiser folder (2025 training files + final_ranking / final_submitting as ranking / submitting.parquet)
FINAL_ORGANISER ?=

.PHONY: help install notebook smoke full full-hedge features final test remove-runs

help:
	@echo "make install      install the pinned dependencies (Python 3.12)"
	@echo "make notebook     regenerate notebooks/v77_pipeline.ipynb from src/v77/make_notebook.py"
	@echo "make smoke        every stage on 5% of the days, small models (~13 min with a GPU)"
	@echo "make full         train everything on the pinned feature table (~12.5 h on the GPU)"
	@echo "make full-hedge   the same with the v77 Rome hedge options"
	@echo "make features     rebuild the feature table from the raw inputs (stage 1 from zero), then train"
	@echo "make final        rebuild the February/June rows of the final file and check it (FINAL_ORGANISER=<folder>)"
	@echo "make test         pre-training checks on a real data slice (~10 min)"

install:
	$(PY) -m pip install -r requirements.txt

notebook:
	$(PY) src/v77/make_notebook.py

smoke: notebook
	$(PY) src/v77/run_notebook.py --smoke

full: notebook
	$(PY) src/v77/run_notebook.py

full-hedge: notebook
	$(PY) src/v77/run_notebook.py --hedge

features: notebook
	$(PY) src/v77/run_notebook.py --features build

final:
	$(PY) src/v77/final_compose.py $(if $(FINAL_ORGANISER),--organiser $(FINAL_ORGANISER),)

test:
	$(PY) -m pytest tests/test_pretrain.py -s -q

remove-runs:
	rm -rf results/v77/*_smoke results/v77/executed_smoke_*.ipynb catboost_info
