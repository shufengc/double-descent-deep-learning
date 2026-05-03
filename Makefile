PYTHON ?= python3
XDG_CACHE_HOME ?= $(CURDIR)/.cache
MPLCONFIGDIR ?= $(CURDIR)/.cache/matplotlib
export MPLCONFIGDIR
export XDG_CACHE_HOME

.PHONY: figures paper-figures activation-ablation activation-ablation-smoke

figures:
	$(PYTHON) scripts/make_figures.py

paper-figures: figures

activation-ablation:
	$(PYTHON) -m src.experiments.exp_activation_ablation --device cuda

activation-ablation-smoke:
	$(PYTHON) -m src.experiments.exp_activation_ablation --smoke --device cpu
