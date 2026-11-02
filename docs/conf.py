"""Build the local API reference from the installed package."""

project = "QA_max_cut_lib"
release = "0.1.0a1"
language = "ru"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.mathjax", "numpydoc"]
numpydoc_show_class_members = False
html_theme = "alabaster"
html_title = "QA_max_cut_lib 0.1.0a1"
exclude_patterns = ["_build"]
autodoc_member_order = "bysource"
