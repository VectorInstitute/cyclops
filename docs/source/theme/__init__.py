"""
Sphinx theme packaging.

See http://www.sphinx-doc.org/en/stable/theming.html#distribute-your-theme-as-a-python-package.
"""
from pathlib import Path


package_dir = Path(__file__).resolve().parent
version_path = package_dir / "VERSION"

with version_path.open(encoding = "utf-8") as version_file:
    __version__ = version_file.readline().strip()


def setup(app):
    app.add_html_theme('nextstrain-sphinx-theme', str(package_dir))
    app.setup_extension('sphinx_copybutton')
    # customize sphinx_copybutton https://sphinx-copybutton.readthedocs.io/en/latest/use.html
    app.config['copybutton_prompt_text'] = '$ '
    app.config['copybutton_line_continuation_character'] = '\\'
