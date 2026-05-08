PYTHON ?= python3
PANDOC ?= pandoc
PDF_ENGINE ?= tectonic
XDG_CACHE_HOME ?= $(CURDIR)/.cache
MPLCONFIGDIR ?= $(CURDIR)/.cache/matplotlib
export MPLCONFIGDIR
export XDG_CACHE_HOME

.PHONY: figures paper-figures pdf clean-pdf paper clean-paper

figures:
	$(PYTHON) scripts/make_figures.py

paper-figures: figures

# Build the submission PDF from report.md using pandoc + tectonic.
# Requires:
#   brew install pandoc tectonic poppler   # poppler is optional (pdfinfo for verify)
# pandoc-header.tex maps Unicode math/box chars to LaTeX so the default Latin
# Modern fonts render report.md cleanly; it also disables LaTeX's auto figure
# numbering so the manual "Figure N:" labels in markdown are not double-prefixed.
pdf: report.pdf

report.pdf: report.md pandoc-header.tex
	$(PANDOC) report.md -o report.pdf \
	    --pdf-engine=$(PDF_ENGINE) \
	    --toc --toc-depth=2 \
	    -V geometry:margin=1in \
	    -V monofont="Menlo" \
	    --include-in-header=pandoc-header.tex

clean-pdf:
	rm -f report.pdf

# Build the ACM-sigconf submission paper from paper/main.tex.
# Multi-file structure: main.tex (driver) + abstract.tex + report.tex + references.bib.
# Uses tectonic, which auto-fetches LaTeX packages and runs BibTeX.
paper: paper/main.pdf

paper/main.pdf: paper/main.tex paper/abstract.tex paper/report.tex paper/references.bib
	cd paper && tectonic --keep-intermediates --reruns 3 main.tex

clean-paper:
	cd paper && rm -f main.pdf main.aux main.bbl main.blg main.log main.out main.synctex.gz
