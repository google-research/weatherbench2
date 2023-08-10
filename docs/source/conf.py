# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import subprocess
import sys

import weatherbench2  # verify this works

print("python exec:", sys.executable)
print("sys.path:", sys.path)
print("pip environment:")
subprocess.run([sys.executable, "-m", "pip", "list"])

print(f"xarray_beam: {weatherbench2.__version__}, {weatherbench2.__file__}")

# -- Project information -----------------------------------------------------

project = "WeatherBench 2"
copyright = "2023, Google, LLC"
author = "Weatherbench authors"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "myst_nb",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_logo = "_static/wb2-logo-wide.png"
html_theme_options = {
    "logo_only": True,
    "style_nav_header_background": "white",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

autosummary_generate = True

# https://stackoverflow.com/a/66295922/809705
autodoc_typehints = "description"

nb_kernel_rgx_aliases = {"conda-env-weatherbench2-py": "python3"}
nb_execution_mode = "off"
