# 编译期末报告
# 方式一：make          （需 xelatex + bibtex + ctex）
# 方式二：make tectonic （自动下载宏包，推荐）
MAIN = report
LATEX = xelatex
BIBTEX = bibtex
TECTONIC ?= tectonic

.PHONY: all clean figures tectonic

all: $(MAIN).pdf

tectonic:
	$(TECTONIC) $(MAIN).tex

figures:
	cd .. && python -m experiments.run_all
	cd .. && python -m experiments.extended_experiments

$(MAIN).pdf: $(MAIN).tex preamble.tex references.bib sections/*.tex appendices/*.tex
	$(LATEX) -interaction=nonstopmode $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(LATEX) -interaction=nonstopmode $(MAIN).tex
	$(LATEX) -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).pdf
