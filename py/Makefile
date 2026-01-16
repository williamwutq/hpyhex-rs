# Makefile for Sphinx and Maturin

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
PYVERSION	  ?= python3.11

# Put it first so that "make" without argument is like "make help".
help:
	@echo "Makefile for building Rust extension and Sphinx documentation"
	@echo ""
	@echo "Available targets:"
	@echo "  build      Build the Rust extension module using maturin"
	@echo "  html       Build HTML documentation"
	@echo "  latex      Build LaTeX documentation"
	@echo "  man		Build man page documentation"
	@echo "  clean      Clean up build files"

.PHONY: help Makefile

# Build the Rust extension module
build:
	maturin build --release --interpreter "$(PYVERSION)" --out dist/

# Compile docs for HTML
html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Compile docs for LaTeX
latex:
	@$(SPHINXBUILD) -M latex "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Compile docs for man pages
man:
	@$(SPHINXBUILD) -M man "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Cleanup build files
clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	rm -rf dist/