# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


# -- Project information -----------------------------------------------------
#
# Modified by the ContextPilot project in 2026.

project = "verl"
copyright = "2024 ByteDance Seed Foundation MLSys Team"
author = "Guangming Sheng, Chi Zhang, Yanghua Peng, Haibin Lin"


master_doc = "index"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
napoleon_google_docstring = True
napoleon_numpy_docstring = False

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]

language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]



html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]

html_js_files = [
    "js/runllm-widget.js",
    "js/resizable-sidebar.js",
]

html_css_files = [
    "custom.css",
]

exclude_patterns += ["README.md", "README_vllm0.7.md"]

suppress_warnings = ["ref.duplicate", "ref.myst"]
