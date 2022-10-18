"""Sphinx theme packaging."""

from pathlib import Path

package_dir = Path(__file__).resolve().parent
version_path = package_dir / "VERSION"

with version_path.open(encoding="utf-8") as version_file:
    __version__ = version_file.readline().strip()


def setup(app):
    """Set up package."""
    app.add_html_theme("cyclops-sphinx-theme", str(package_dir))
    app.setup_extension("sphinx_copybutton")
    app.config["copybutton_prompt_text"] = "$ "
    app.config["copybutton_line_continuation_character"] = "\\"
