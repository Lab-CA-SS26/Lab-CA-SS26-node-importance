# Makefile for compiling the lab report

MAIN = main
OUTDIR = out
LATEXMK = latexmk

# Default target
.PHONY: all
all: $(OUTDIR)/$(MAIN).pdf

# Generate the instance table if it doesn't exist
tables/instance_table_new.tex:
	@echo "Generating instance table..."
	cd .. && ./venv/bin/python scripts/generate_instance_table.py

# Rule to compile the main document
$(OUTDIR)/$(MAIN).pdf: $(MAIN).tex tables/instance_table_new.tex
	@mkdir -p $(OUTDIR)
	$(LATEXMK) -pdf -outdir=$(OUTDIR) -interaction=nonstopmode $(MAIN).tex

# Clean auxiliary files and the output directory
.PHONY: clean
clean:
	$(LATEXMK) -C -outdir=$(OUTDIR) $(MAIN).tex
	rm -rf $(OUTDIR)
