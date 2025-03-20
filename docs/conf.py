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
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# sys.path.insert(0, os.path.abspath(os.path.join('..', 'cryoswath')))
# print(sys.path)


# -- Project information -----------------------------------------------------

project = 'cryoswath'
copyright = '2024-%Y, Jan Haacker'
author = 'Jan Haacker'

# The full version, including alpha/beta/rc tags
release = '0.2.1-a630abb'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx.ext.linkcode',
    # 'myst_parser'
]


# for extension linkcode to work
# copied from numpy
def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    if domain != 'py':
        return None

    import inspect
    import cryoswath

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    # strip decorators, which would resolve to the source of the decorator
    # possibly an upstream bug in getsourcefile, bpo-1764286
    try:
        unwrap = inspect.unwrap
    except AttributeError:
        pass
    else:
        obj = unwrap(obj)

    fn = None
    lineno = None

    if fn is None:
        try:
            fn = inspect.getsourcefile(obj)
        except Exception:
            fn = None
        if not fn:
            return None

        # Ignore re-exports as their source files are not within the cryoswath repo
        module = inspect.getmodule(obj)
        if module is not None and not module.__name__.startswith("cryoswath"):
            return None

        try:
            source, lineno = inspect.getsourcelines(obj)
        except Exception:
            lineno = None

        fn = os.path.relpath(fn, start=os.path.dirname(cryoswath.__file__))

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    # maybe restructure in future: have doc "stable" point to main, "latest" to develop
    if 'dev' in cryoswath.__version__:
        return "https://github.com/j-haacker/cryoswath/blob/develop/cryoswath/%s%s" % (
           fn, linespec)  # should actually refer to "main"
    else:
        return "https://github.com/j-haacker/cryoswath/blob/v%s/cryoswath/%s%s" % (
           cryoswath.__version__, fn, linespec)


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")
