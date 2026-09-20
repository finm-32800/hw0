"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based.

Run `doit` from the root of the repository to execute the notebook and save an
HTML copy of it in `_output/`. We will cover `doit` in depth later in the course.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(1, "./src/")

import config

OUTPUT_DIR = Path(config.OUTPUT_DIR)
MANUAL_DATA_DIR = Path(config.MANUAL_DATA_DIR)


def move(origin, destination):
    shutil.move(origin, destination)


def task_run_notebook():
    """Convert the notebook from its .py source, execute it, and export HTML."""
    pyfile = Path("./src/01_markowitz.ipynb.py")
    notebook = pyfile.with_suffix("")  # strips .py, leaves .ipynb
    return {
        "actions": [
            (OUTPUT_DIR.mkdir, [], {"parents": True, "exist_ok": True}),
            f"jupytext --to notebook --output {notebook} {pyfile}",
            # Executed in place so that the notebook runs from inside `src/`,
            # where it can import the other modules.
            f"jupyter nbconvert --execute --to notebook --inplace {notebook}",
            f"jupyter nbconvert --to html --output-dir={OUTPUT_DIR} {notebook}",
            (move, [notebook, OUTPUT_DIR / notebook.name]),
        ],
        "file_dep": [
            pyfile,
            "./src/pull_crsp.py",
            MANUAL_DATA_DIR / "crsp_monthly_returns.csv",
        ],
        "targets": [OUTPUT_DIR / notebook.name, OUTPUT_DIR / "01_markowitz.html"],
        "clean": True,
    }
