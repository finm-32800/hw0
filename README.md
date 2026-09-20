# HW 0: Portfolio Selection (Markowitz, 1952)

This is the practice homework for *FINM 32800: Data Pipelines for Quantitative
Research*. It is ungraded. It accompanies
[Lecture 0 in the course textbook](https://finm-32800.github.io/overview_w0.html),
and the full instructions are on the
[HW 0 page](https://finm-32800.github.io/HW0.html).

In one small project you will see the pattern that the rest of the course
repeats at larger scale: a data pull, an analysis in a notebook, the same logic
moved into tested functions, and an automated check that runs every time you
push to GitHub.

## Quick start

```bash
git clone https://github.com/YOUR-USERNAME/hw0.git
cd hw0
conda create -n finm python=3.12
conda activate finm
pip install -r requirements.txt
```

Then check that everything works:

```bash
streamlit run src/app.py   # interactive dashboard
pytest                     # unit tests (they fail until you finish the homework)
```

## Getting the data

The notebook and the dashboard use a small extract of monthly stock returns
from CRSP. CRSP data is licensed, so the extract is not stored in this public
repository. Download `crsp_monthly_returns.csv` from the link shared in lecture
and on Canvas, and save it as

```
data_manual/crsp_monthly_returns.csv
```

Without the file, the dashboard still runs using simulated returns, and the
unit tests do not need it at all.

The file `src/pull_crsp.py` is the exact code that produced the extract. You do
not need to run it, and you cannot until your WRDS account is approved. Read
it anyway: every number in a reproducible pipeline should trace back to code.

## What to do

1. Read and run the notebook, `src/01_markowitz.ipynb.py`. It is a Python
   script in the "percent" format. VS Code can run it cell by cell, or you can
   build an executed copy in `_output/` by running `doit`.
2. Fill in the two functions marked `TODO` in `src/port_opt.py`.
3. Complete the three GitHub Skills tutorials listed on the HW 0 page and
   record them in `src/github_skills.py`.
4. Run `pytest`. When the tests pass, commit and push. The same tests run
   automatically on GitHub Actions; look for the green check mark next to your
   commit.

Do not edit the test files (`src/test_*.py`).

## What is in this repository

| Path | Purpose |
|---|---|
| `requirements.txt` | The exact package versions this project was tested with |
| `src/pull_crsp.py` | Pulls the data from WRDS (for reference) |
| `src/01_markowitz.ipynb.py` | The notebook: mean-variance analysis on real data |
| `src/port_opt.py` | The functions you complete |
| `src/test_port_opt.py`, `src/test_github_skills.py` | Unit tests |
| `src/app.py` | Streamlit dashboard |
| `src/config.py` | Paths and settings, read from an optional `.env` file |
| `dodo.py` | Task runner file (`doit`), covered later in the course |
| `.github/workflows/tests.yml` | Runs the tests on GitHub Actions on every push |
| `data_manual/` | Data that cannot be recreated automatically |
| `_data/`, `_output/` | Generated files. Safe to delete; never committed |
