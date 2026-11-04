"""Build the local API reference from the installed package."""

from QA_max_cut_lib._version import __version__

project = "QA_max_cut_lib"
release = __version__
language = "en"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "numpydoc",
    "myst_parser",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["dollarmath", "colon_fence", "deflist"]
numpydoc_show_class_members = False
html_theme = "alabaster"
html_title = f"QA_max_cut_lib {__version__}"
exclude_patterns = ["_build"]
autodoc_member_order = "bysource"
