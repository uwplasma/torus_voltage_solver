from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

project = "torus-solver"
copyright = "2026, torus-solver contributors"

try:
    release = importlib.metadata.version("torus-solver")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    release = "0.0.0"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": True}
autodoc_typehints = "description"

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "strikethrough",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

html_theme = "furo"
html_title = project
html_static_path = ["_static"]
html_css_files = ["custom.css"]
