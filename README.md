# Potential Talent Finder

## Overview

This project helps recruiters surface promising candidates for specific roles when no historical hiring labels are available. It relies on an interpretable heuristic ranking model driven by:
- Title similarity to recruiter-provided keywords.
- Normalised connection counts (e.g. `500+`).
- Optional location preferences.
- Optional feedback from starring already-reviewed candidates.

You can explore the logic in the notebook, call the ranking functions directly, or interact with a Streamlit app that works with the bundled demo data or your own CSV.

## Data Schema

The ranking pipeline expects a tabular dataset with the following columns:
- `id` – unique candidate identifier (numeric or string acceptable).
- `job_title` – candidate headline or title text.
- `location` – free-text location string.
- `connection` – connection count as text (e.g. `85`, `500+`).

Any additional columns are ignored.

## Components

- `heuristic_ranking.py` – reusable module implementing the scoring logic (`RankingConfig`, `rank_candidates`).
- `Potential_Talents 1.2.ipynb` – notebook illustrating baseline ranking plus re-ranking after starring a candidate.
- `app.py` – Streamlit interface for non-technical reviewers; supports keyword filters, location/connection constraints, optional starring, and user-supplied CSVs.
- `potential-talents - Aspiring human resources - seeking human resources.csv` – demo dataset used across examples.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # or pip install pandas numpy scikit-learn streamlit altair
```

If you do not maintain a requirements file, install at minimum:
`pandas`, `numpy`, `scikit-learn`, `streamlit` (for the UI), and optionally `altair` for advanced charts.

## Usage

### 1. Programmatic ranking

```python
from heuristic_ranking import RankingConfig, rank_candidates

config = RankingConfig(
    keywords=["aspiring human resources", "seeking human resources"],
    preferred_locations=["Raleigh-Durham, North Carolina Area"],
    starred_ids=[17],  # optional
)
ranked = rank_candidates(
    "potential-talents - Aspiring human resources - seeking human resources.csv",
    config,
)
print(ranked.head())
```

- Pass either a path to a CSV or a prepared `pandas.DataFrame`.
- Tune weights (`weight_similarity`, `weight_connections`, `weight_location`) if you want to emphasise different signals.

### 2. Notebook walkthrough

Open `Potential_Talents 1.2.ipynb` in Jupyter or VS Code:
1. Run the initial ranking cell to produce the shortlist.
2. Provide a starred candidate to see how the ordering adapts.

### 3. Streamlit app

```bash
streamlit run app.py
```

Within the sidebar you can:
- Choose between the bundled demo dataset or upload your own CSV with the required columns.
- Enter search keywords (one per line).
- Filter by preferred locations and minimum connection count.
- Expand the “Advanced & feedback” section to tweak weights or star candidate IDs.

The main panel only presents the ranked candidates so reviewers focus on outcomes, not internal scoring details.

## Using Your Own Dataset

1. Prepare a CSV containing the four required columns (`id`, `job_title`, `location`, `connection`).
2. Launch the Streamlit app, pick **Upload CSV**, and select your file.
3. Alternatively, load the data via `pandas` and call `rank_candidates(dataframe, config)` directly.

If a required column is missing, both the app and the module raise a descriptive error.

## Future Enhancements

- Collect reviewer or hiring outcomes to transition from heuristics to a supervised ranking model.
- Incorporate richer features (skills, tenure, seniority) for finer-grained differentiation.
- Evaluate recommendation quality with ranking metrics (MAP, nDCG) once labelled data is available.
- Automate feedback loops so repeated starring actions continually update model weights.
