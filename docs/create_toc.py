import os
from pathlib import Path

# Copy 01-about/000_intro.md to index.md
# We want the intro to the first chapter to also be the intro to the book,
# and it doesn't seem to be easily possible to get rid of an index file entirely.
directory = Path(__file__).parent
with open(directory / "01-about/000_intro.md", "r", encoding="utf-8") as src:
    intro_content = src.read()
with open(directory / "index.md", "w", encoding="utf-8") as dst:
    dst.write("<!-- This is a copy of 01-about/000_intro.md; DO NOT EDIT -->\n\n")
    dst.write(intro_content)

# This script provides a platform-independent way of making the jupyter-book call (used in pyproject.toml)
folder = Path(__file__).parent
toc_file = folder / "_toc.yml"
cmd = f"jupyter-book toc from-project docs -e .rst -e .md -e .ipynb >{toc_file}"
print(cmd)
os.system(cmd)
