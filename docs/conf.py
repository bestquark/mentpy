# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
import mentpy

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "mentpy"
copyright = "2022, Luis Mantilla"
author = "Luis Mantilla"

# The short X.Y version.
version = ".".join(str(x) for x in mentpy.__version_info__[:2])
# The full version, including alpha/beta/rc tags.
release = mentpy.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_immaterial",
    "sphinx_immaterial.apidoc.python.apigen",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
]


# -- Sphinx Immaterial configs -------------------------------------------------

python_apigen_modules = {
    "mentpy": "api/mentpy.",
}

python_apigen_default_groups = [
    ("class:.*", "Classes"),
    ("data:.*", "Variables"),
    ("function:.*", "Functions"),
    ("method:.*", "Methods"),
    ("classmethod:.*", "Class methods"),
    ("property:.*", "Properties"),
    (r"method:.*\.[A-Z][A-Za-z,_]*", "Constructors"),
    (r"method:.*\.__[A-Za-z,_]*__", "Special methods"),
    (r"method:.*\.__(init|new)__", "Constructors"),
    (r"method:.*\.__(str|repr)__", "String representation"),
    # (r"method:.*\.is_[a-z,_]*", "Tests"),
    # (r"property:.*\.is_[a-z,_]*", "Tests"),
]
python_apigen_default_order = [
    ("class:.*", 10),
    ("data:.*", 11),
    ("function:.*", 12),
    ("method:.*", 24),
    ("classmethod:.*", 22),
    ("property:.*", 30),
    (r"method:.*\.[A-Z][A-Za-z,_]*", 20),
    (r"method:.*\.__[A-Za-z,_]*__", 23),
    (r"method:.*\.__(init|new)__", 20),
    (r"method:.*\.__(str|repr)__", 21),
    # (r"method:.*\.is_[a-z,_]*", 31),
    # (r"property:.*\.is_[a-z,_]*", 32),
]

python_apigen_order_tiebreaker = "alphabetical"
python_apigen_case_insensitive_filesystem = False
python_apigen_show_base_classes = True

python_transform_type_annotations_pep585 = False

object_description_options = [
    ("py:.*", dict(include_rubrics_in_toc=True)),
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_immaterial"
html_static_path = ["_static"]

html_title = "MentPy"
# html_favicon = "../logo/mentpy-favicon-color.png"
# html_logo = "../logo/mentpy-favicon-white.png"


# html_theme_options = {
#     "icon": {
#         "repo": "fontawesome/brands/github",
#     },
#     "site_url": "https://mentpy.readthedocs.io/",
#     "repo_url": "https://github.com/bestquark/mentpy",
#     "repo_name": "bestquark/mentpy",
#     "repo_type": "github",
#     "social": [
#         {
#             "icon": "fontawesome/brands/github",
#             "link": "https://github.com/bestquark/mentpy"
#         },
#         {
#             "icon": "fontawesome/brands/python",
#             "link": "https://pypi.org/project/mentpy/"
#         },
#         {
#             "icon": "fontawesome/brands/twitter",
#             "link": "https://twitter.com/mentpy"
#         },
#     ],
#     "edit_uri": "",
#     "globaltoc_collapse": False,
#     "features": [
#         # "navigation.expand",
#         "navigation.tabs",
#         # "toc.integrate",
#         # "navigation.sections",
#         # "navigation.instant",
#         # "header.autohide",
#         "navigation.top",
#         "navigation.tracking",
#         "toc.follow",
#         "toc.sticky"
#     ],
#     "palette": [
#         {
#             "media": "(prefers-color-scheme: light)",
#             "scheme": "default",
#             "accent": "deep-orange",
#             "toggle": {
#                 "icon": "material/weather-night",
#                 "name": "Switch to dark mode",
#             },
#         },
#         {
#             "media": "(prefers-color-scheme: dark)",
#             "scheme": "slate",
#             "accent": "deep-orange",
#             "toggle": {
#                 "icon": "material/weather-sunny",
#                 "name": "Switch to light mode",
#             },
#         },
#     ],
#     "analytics": {
#         "provider": "google",
#         "property": "G-XXX"
#     },
#     "version_dropdown": True,
#     "version_json": "../versions.json",
# }

# -- Extension configuration -------------------------------------------------

# Create hyperlinks to other documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

autodoc_default_options = {
    "imported-members": True,
    "members": True,
    # "special-members": True,
    # "inherited-members": "ndarray",
    # "member-order": "groupwise",
}
autodoc_typehints = "signature"
autodoc_typehints_description_target = "documented"
autodoc_typehints_format = "short"
myst_enable_extensions = ["dollarmath"]

ipython_execlines = ["import math", "import numpy as np", "import mentpy"]
