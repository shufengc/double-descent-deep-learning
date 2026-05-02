PYTHON ?= python3
XDG_CACHE_HOME ?= $(CURDIR)/.cache
MPLCONFIGDIR ?= $(CURDIR)/.cache/matplotlib
export MPLCONFIGDIR
export XDG_CACHE_HOME

.PHONY: figures paper-figures

figures:
	$(PYTHON) scripts/make_figures.py

paper-figures: figures
