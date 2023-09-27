# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import mentpy

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "mentpy"
copyright = "2022-2023, Luis Mantilla"
author = "Luis Mantilla"

# The short X.Y version.
version = ".".join(str(x) for x in mentpy.__version_info__[:2])
# The full version, including alpha/beta/rc tags.
release = mentpy.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_immaterial",
    "sphinx_immaterial.apidoc.python.apigen",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
]


# -- Latex configuration -----------------------------------------------------
latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    "preamble": r"""
\usepackage{amssymb}
""",
}

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
html_favicon = "_static/favicon-32x32.png"
html_logo = "_static/logo-state.png"

html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/github",
    },
    "site_url": "https://github.com/BestQuark/mentpy",
    "repo_url": "https://github.com/BestQuark/mentpy",
    "repo_name": "MentPy",
    "repo_type": "github",
    # "google_analytics": ["UA-XXXXX", "auto"],
    "globaltoc_collapse": True,
    "features": [
        # "navigation.expand",
        "navigation.tabs",
        # "toc.integrate",
        # "navigation.sections",
        # "navigation.instant",
        # "header.autohide",
        "navigation.top",
        # "navigation.tracking",
        # "search.highlight",
        "search.share",
        "toc.follow",
        "toc.sticky",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "indigo",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "indigo",
            "toggle": {
                "icon": "material/lightbulb",
                "name": "Switch to light mode",
            },
        },
    ],
    # BEGIN: version_dropdown
    "version_dropdown": False,
    # END: version_dropdown
    "toc_title_is_page_title": True,
}

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

ipython_execlines = [
    "import math",
    "import numpy as np",
    "import networkx as nx",
    "import matplotlib.pyplot as plt",
    "import mentpy as mp",
]

rst_prolog = """
    .. role:: python(code)
        :language: python
"""
