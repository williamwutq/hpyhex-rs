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
	@echo "  build-nonp Build the Rust extension module without NumPy support"
	@echo "  build-half Build the Rust extension module with half-precision support"
	@echo "  build-all  Build the Rust extension module for all supported Python versions"
	@echo "  html       Build HTML documentation"
	@echo "  latex      Build LaTeX documentation"
	@echo "  man		Build man page documentation"
	@echo "  clean      Clean up build files"
	@echo "  release    Release the package to PyPI using twine"

.PHONY: help Makefile

# Build the Rust extension module
build:
	maturin build --release --interpreter "$(PYVERSION)" --out dist/ --features numpy

# Build the Rust extension module without NumPy support
build-nonp:
	maturin build --release --interpreter "$(PYVERSION)" --out dist/ --features default

# Build the Rust extension module with half-precision support
build-half:
	maturin build --release --interpreter "$(PYVERSION)" --out dist/ --features full

# Build for all supported python versions
build-all:
	@maturin build --release --interpreter python3.8 --out dist/ --features numpy
	@maturin build --release --interpreter python3.9 --out dist/ --features numpy
	@maturin build --release --interpreter python3.10 --out dist/ --features numpy
	@maturin build --release --interpreter python3.11 --out dist/ --features numpy
	@maturin build --release --interpreter python3.12 --out dist/ --features numpy

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

# Release the package
release:
	python -m twine upload dist/*  
