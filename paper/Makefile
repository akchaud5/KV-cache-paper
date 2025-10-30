# Makefile for MLSys 2025 Paper

PAPER = mlsys_kv_compression
TEX = $(PAPER).tex
PDF = $(PAPER).pdf
BIB = references.bib

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Compilation flags
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error

.PHONY: all paper clean distclean view

# Default target
all: paper

# Build the paper
paper: $(PDF)

$(PDF): $(TEX) $(BIB)
	@echo "Building paper..."
	$(LATEX) $(LATEX_FLAGS) $(TEX)
	$(BIBTEX) $(PAPER)
	$(LATEX) $(LATEX_FLAGS) $(TEX)
	$(LATEX) $(LATEX_FLAGS) $(TEX)
	@echo "Build complete: $(PDF)"

# Quick build (no bibtex)
quick:
	@echo "Quick build (no bibliography update)..."
	$(LATEX) $(LATEX_FLAGS) $(TEX)
	@echo "Quick build complete: $(PDF)"

# Clean auxiliary files
clean:
	@echo "Cleaning auxiliary files..."
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
	@echo "Clean complete"

# Remove all generated files including PDF
distclean: clean
	@echo "Removing PDF..."
	rm -f $(PDF)
	@echo "Distclean complete"

# View the PDF (requires a PDF viewer)
view: $(PDF)
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(PDF); \
	elif command -v open > /dev/null; then \
		open $(PDF); \
	elif command -v evince > /dev/null; then \
		evince $(PDF); \
	else \
		echo "No PDF viewer found. Please open $(PDF) manually."; \
	fi

# Check for common LaTeX errors
check:
	@echo "Checking for common issues..."
	@grep -n "TODO\|FIXME\|XXX" $(TEX) || echo "No TODOs found"
	@echo "Check complete"

# Word count (approximate)
wordcount:
	@echo "Approximate word count:"
	@texcount -brief $(TEX)

# Spell check (requires aspell)
spell:
	@echo "Running spell check..."
	@aspell -t -c $(TEX)

# Help message
help:
	@echo "MLSys Paper Makefile"
	@echo "===================="
	@echo "Available targets:"
	@echo "  make          - Build the paper (default)"
	@echo "  make paper    - Build the paper with bibliography"
	@echo "  make quick    - Quick build without bibliography update"
	@echo "  make clean    - Remove auxiliary files"
	@echo "  make distclean- Remove all generated files including PDF"
	@echo "  make view     - Open the PDF in default viewer"
	@echo "  make check    - Check for TODOs and common issues"
	@echo "  make wordcount- Count words in the paper"
	@echo "  make spell    - Run spell checker"
	@echo "  make help     - Show this help message"
