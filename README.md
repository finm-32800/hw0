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

```bash
doit
```

`doit` reads the file `dodo.py`, gets the data, and then executes the notebooks,
saving HTML copies of them in `_output/`. The data is a small extract of monthly
stock returns from CRSP, saved as `_data/crsp_monthly_returns.csv`.

There are two ways to get it, and `src/pull_crsp.py` implements both:

- **By default,** it downloads a cached copy of the extract. This needs no
  account, so it works before your WRDS access is approved.
- **With `NO_CACHE=True`,** it pulls the data fresh from WRDS. Copy
  `.env.example` to `.env`, fill in your `WRDS_USERNAME`, set `NO_CACHE=True`,
  and run `doit` again.

Either way, read the WRDS queries in `src/pull_crsp.py`. Every number in a
reproducible pipeline should trace back to code, and those queries are the code
that produced the cached copy.

Without the data file, the dashboard still runs using simulated returns, and
the unit tests do not need it at all.

## What to do

1. Read and run the notebook, `src/01_markowitz.ipynb.py`. It is a Python
   script in the "percent" format. VS Code can run it cell by cell, and `doit`
   builds an executed copy in `_output/`. The appendix notebook,
   `src/02_markowitz_derivation.ipynb.py`, derives the formulas.
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
| `src/pull_crsp.py` | Gets the data: a cached download by default, WRDS with `NO_CACHE=True` |
| `src/01_markowitz.ipynb.py` | The notebook: mean-variance analysis on real data |
| `src/02_markowitz_derivation.ipynb.py` | Appendix notebook: derivation of the formulas |
| `src/mean_variance.py` | Frontier calculations, with and without short sales |
| `src/port_opt.py` | The functions you complete |
| `src/test_*.py` | Unit tests |
| `src/app.py` | Streamlit dashboard |
| `src/config.py` | Paths and settings, read from an optional `.env` file |
| `dodo.py` | Task runner file (`doit`): gets the data and builds the notebooks |
| `.github/workflows/tests.yml` | Runs the tests on GitHub Actions on every push |
| `_data/`, `_output/` | Downloaded data and generated files. Safe to delete; never committed |
| `data_manual/` | For data that cannot be recreated automatically (unused here) |
