# Makefile for compiling the thesis report

MAIN = main
OUTDIR = out
LATEXMK = latexmk

# Default target
.PHONY: all
all: $(OUTDIR)/$(MAIN).pdf

# Rule to compile the main document
$(OUTDIR)/$(MAIN).pdf: $(MAIN).tex
	@mkdir -p $(OUTDIR)
	$(LATEXMK) -pdf -outdir=$(OUTDIR) -interaction=nonstopmode $(MAIN).tex

# Clean auxiliary files and the output directory
.PHONY: clean
clean:
	$(LATEXMK) -C -outdir=$(OUTDIR) $(MAIN).tex
	rm -rf $(OUTDIR)
