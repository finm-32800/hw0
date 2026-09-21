"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based.

Run `doit` from the root of the repository. It gets the data, then executes the
notebooks and saves HTML copies of them in `_output/`. Run `doit list` to see
the tasks. We will cover `doit` in depth later in the course.

By default the data task downloads a cached extract. To pull the data fresh
from WRDS instead, set NO_CACHE=True, either in your `.env` file or on the
command line:

    NO_CACHE=True doit
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(1, "./src/")

from doit.tools import config_changed

import config

DATA_DIR = Path(config.DATA_DIR)
OUTPUT_DIR = Path(config.OUTPUT_DIR)
EXTRACT = DATA_DIR / "crsp_monthly_returns.csv"


def move(origin, destination):
    shutil.move(origin, destination)


def task_pull_data():
    """Get the CRSP extract: a cached download by default, WRDS if NO_CACHE=True."""
    return {
        "actions": ["python ./src/pull_crsp.py"],
        "file_dep": ["./src/pull_crsp.py"],
        "targets": [EXTRACT],
        # Rerun if the choice of data source changes.
        "uptodate": [config_changed(str(config.NO_CACHE))],
        "verbosity": 2,
        "clean": True,
    }


def task_run_notebooks():
    """Convert each notebook from its .py source, execute it, and export HTML."""
    for stem in ["01_markowitz", "02_markowitz_derivation"]:
        pyfile = Path(f"./src/{stem}.ipynb.py")
        notebook = pyfile.with_suffix("")  # strips .py, leaves .ipynb
        yield {
            "name": stem,
            "actions": [
                (OUTPUT_DIR.mkdir, [], {"parents": True, "exist_ok": True}),
                f"jupytext --to notebook --output {notebook} {pyfile}",
                # Executed in place so that the notebook runs from inside
                # `src/`, where it can import the other modules.
                f"jupyter nbconvert --execute --to notebook --inplace {notebook}",
                f"jupyter nbconvert --to html --output-dir={OUTPUT_DIR} {notebook}",
                (move, [notebook, OUTPUT_DIR / notebook.name]),
            ],
            "file_dep": [pyfile, "./src/mean_variance.py", "./src/pull_crsp.py", EXTRACT],
            "targets": [OUTPUT_DIR / notebook.name, OUTPUT_DIR / f"{stem}.html"],
            "clean": True,
        }
