# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from sphinxawesome_theme.postprocess import Icons

html_permalinks_icon = Icons.permalinks_icon  # SVG as a string

project = "Linear Operator Learning"
copyright = "2025, Linear Operator Learning Team"
author = "Linear Operator Learning Team"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxawesome_theme",
    "myst_parser",
    "sphinxcontrib.bibtex",
    "sphinx_design"
]

myst_enable_extensions = ["amsmath", "dollarmath", "html_image"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

bibtex_bibfiles = ["bibliography.bib"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinxawesome_theme"
autodoc_class_signature = "separated"
autoclass_content = "class"

autodoc_typehints = "description"
autodoc_member_order = "groupwise"
napoleon_preprocess_types = True
napoleon_use_rtype = False
autodoc_mock_imports = ["escnn", "escnn.group", "symm_torch"]

master_doc = "index"

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "restructuredtext",
    ".md": "markdown",
}

# Favicon configuration
# html_favicon = '_static/favicon.ico'

# Configure syntax highlighting for Awesome Sphinx Theme
pygments_style = "default"
pygments_style_dark = "material"

# Additional theme configuration
html_title = "Linear Operator Learning"
html_theme_options = {
    "show_prev_next": False,
    "show_scrolltop": True,
    "extra_header_link_icons": {
        "GitHub": {
            "link": "https://github.com/CSML-IIT-UCL/linear_operator_learning",
            "icon": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" height="18" fill="currentColor"><path d="M12 0C5.373 0 0 5.373 0 12c0 5.302 3.438 9.8 8.205 11.387.6.111.82-.261.82-.577 0-.285-.01-1.04-.015-2.04-3.338.726-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.757-1.333-1.757-1.09-.745.083-.729.083-.729 1.205.084 1.84 1.238 1.84 1.238 1.07 1.835 2.807 1.305 3.492.998.108-.775.418-1.305.76-1.605-2.665-.305-5.466-1.332-5.466-5.93 0-1.31.467-2.38 1.235-3.22-.125-.303-.535-1.523.115-3.176 0 0 1.005-.322 3.3 1.23.955-.265 1.98-.398 3-.403 1.02.005 2.045.138 3 .403 2.28-1.552 3.285-1.23 3.285-1.23.655 1.653.245 2.873.12 3.176.77.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.62-5.475 5.92.43.37.81 1.1.81 2.22 0 1.605-.015 2.895-.015 3.285 0 .32.215.694.825.575C20.565 21.795 24 17.3 24 12c0-6.627-5.373-12-12-12z"/></svg>""",
        },
    },
    "show_breadcrumbs": True,
}

html_sidebars = {
  "**": ["sidebar_main_nav_links.html", "sidebar_toc.html"]
}

html_css_files = ["custom.css"]

html_favicon = "favicon.ico"
html_static_path = ["_static"]
templates_path = ["_templates"]