# setting the system path
import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# ---------------------------------------------- Project information ---------------------------------------------------
project = 'Karaokey'
copyright = '2019, Vinay Patel'
author = 'Vinay Patel'

version = '0.1.0'
release = 'beta'
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon']

# napoleon extension settings for numpydoc
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

templates_path = ['_templates']

source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'
language = None
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = None
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
htmlhelp_basename = 'Karaokeydoc'

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

latex_documents = [
    (master_doc, 'Karaokey.tex', 'Karaokey Documentation',
     'w1572032 - Vinay Patel', 'manual'),
]

man_pages = [
    (master_doc, 'karaokey', 'Karaokey Documentation',
     [author], 1)
]
texinfo_documents = [
    (master_doc, 'Karaokey', 'Karaokey Documentation',
     author, 'Karaokey', 'Separating vocals from music files using Deep Learning.',
     'Miscellaneous'),
]

epub_title = project
epub_exclude_files = ['search.html']
intersphinx_mapping = {'https://docs.python.org/': None}
todo_include_todos = True
