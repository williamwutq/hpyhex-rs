# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

def fix_rust_crates(app, config):
    rc = config.rust_crates
    if isinstance(rc, list):
        config.rust_crates = {
            entry["name"]: entry["path"]
            for entry in rc
        }

def setup(app):
    app.connect("config-inited", fix_rust_crates)

project = 'hpyhex_rs'
copyright = '2026, William Wu'
author = 'William Wu'
release = '0.2.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinxcontrib_rust",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

rust_crates = [
    {
        "name": "py",
        "path": "../py",
    }
]

rust_doc_dir = "docs/crates/"
rust_rustdoc_fmt = "rst"



myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
