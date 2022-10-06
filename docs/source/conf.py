"""Configuration file for the Sphinx documentation builder."""

# pylint: disable-all

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath("../../cyclops"))


# -- Project information -----------------------------------------------------

project = "cyclops"
copyright = "2022, Vector AI Engineering"
author = "Vector AI Engineering"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx.ext.viewcode",
    "sphinxcontrib.apidoc",
    "myst_parser",
]
numpydoc_show_inherited_class_members = False
numpydoc_show_class_members = False
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
add_module_names = False

apidoc_module_dir = "../../cyclops"
apidoc_excluded_paths = ["tests", "models", "*constants.py", "*column_names.py"]
apidoc_output_dir = "reference/api"
apidoc_separate_modules = True
apidoc_extra_args = ["-f", "-M", "-T", "--implicit-namespaces"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.9/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sqlalchemy": ("https://docs.sqlalchemy.org", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: list = ["reference/api/cyclops.rst"]
source_suffix = [".rst", ".md"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
