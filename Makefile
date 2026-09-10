# Four targets. Everything else is a documented command in the README.
PY ?= python
CONFIG ?= configs/diema7_stgcn_recipe.yaml
SEED   ?= 255
FOLDS  ?= 10

.PHONY: install test test-cov lpo

install:            ## install the package and its dev extras
	$(PY) -m pip install -e ".[dev]"

test:               ## the fast suite: no dataset, no GPU, ~15 s
	$(PY) -m pytest tests/ -m "not slow" -q

test-cov:
	$(PY) -m pytest tests/ -m "not slow" --cov=emo_mocap --cov-report=term-missing

lpo:                ## train every fold for one seed, then score them
	scripts/run_lpo.sh

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
