# 🚗 Used Car Price Predictor — Pakistan

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-live-brightgreen.svg)

**Live app:** [usedcarpricepredictorpak.streamlit.app](https://usedcarpricepredictorpak.streamlit.app/)

Predicts a fair market price for used cars in Pakistan, trained on real
listings scraped from [PakWheels.com](https://www.pakwheels.com) —
Pakistan's largest used car marketplace.

Unlike generic car price predictors (almost all trained on US/European
markets), this model is built around **Pakistan-specific pricing
dynamics** that standard datasets don't capture:

- **Import/assembly status** — locally assembled vs. imported (JDM/duty
  status) cars follow different price curves in Pakistan. This ranked in
  the model's **top 10 most important features**.
- **Registered-in province vs. listing province** — whether a car is
  registered where it's being sold, or brought in from elsewhere.

## Table of Contents

- [Why this project](#why-this-project)
- [Results](#results)
- [Project layout](#project-layout)
- [Setup](#setup)
- [Usage](#usage)
- [What I'd improve next](#what-id-improve-next)
- [Tech stack](#tech-stack)
- [License](#license)

## Why this project

Most "car price prediction" portfolio projects reuse the same handful of
public Kaggle datasets (largely US/German market data) and don't reflect
how used cars are actually priced in Pakistan. This project scrapes
fresh, current listings (~3,900 cars, all scraped directly, not from an
existing dataset) and engineers features specific to the local market,
so the model — and the live "is this a fair price?" tool built on top of
it — actually reflects how Pakistani buyers and sellers price cars.

Full write-up with EDA, feature rationale, and error analysis:
[`notebooks/analysis.ipynb`](notebooks/analysis.ipynb).

## Results

Final model: **XGBoost**, tuned via 5-fold cross-validated randomized
search, evaluated on a held-out 20% test set.

| Model | MAE | MAPE |
|---|---|---|
| Linear Regression (baseline) | Rs. 913,139 | 16.7% |
| XGBoost (default params) | Rs. 579,904 | 10.3% |
| **XGBoost (tuned)** | **Rs. 572,356** | **9.9%** |

**Honest limitations:**
- The model is most reliable for high-volume makes (Toyota, Honda,
  Suzuki — 7.5–9.6% MAPE) and least reliable for rare/luxury makes
  (Nissan, Mitsubishi, Mercedes-Benz, Audi — 17–23% MAPE), simply because
  it saw far fewer examples of them during training.
- The cheapest price quartile has the worst *percentage* error (13.4%
  MAPE) despite the smallest absolute error (~Rs. 173k) — a small rupee
  miss matters proportionally more on an inexpensive car.
- `registered_matches_listing` (the registration-province feature) did
  **not** turn out to be a strong predictor, despite the original
  hypothesis — reported here honestly rather than left out.

Full breakdown, plots, and reasoning in the analysis notebook.

## Project layout

```
car_price_predictor/
├── app/
│   └── app.py              # Streamlit app (live demo)
├── data/
│   ├── raw/                 # scraper output — gitignored (reproducible via scraper.py)
│   └── processed/
│       ├── cars_clean.csv      # gitignored (reproducible via clean.py)
│       └── cars_features.csv   # tracked — the app loads this at runtime
├── scripts/
│   ├── scraper.py            # two-stage scraper (search pages + detail pages)
│   ├── clean.py               # parses model from title, converts types, standardizes
│   ├── features.py            # engineers the Pakistan-specific features
│   ├── train.py                # baseline + XGBoost training
│   └── tune_and_analyze.py     # hyperparameter search + error analysis
├── notebooks/
│   └── analysis.ipynb          # full EDA, results, and error analysis write-up
├── models/
│   ├── xgb_model.json           # tracked — the app loads this at runtime
│   ├── model_metadata.json       # tracked — feature lists, metrics, category levels
│   └── error_analysis.json        # per-make / per-price-bucket error breakdown
├── .github/workflows/lint.yml
├── requirements.txt
├── packages.txt              # system-level deps for Streamlit Cloud (libgomp1 for xgboost)
└── README.md
```

## Setup

```bash
git clone https://github.com/MuhammadAliEjaz1/car_price_predictor.git
cd car_price_predictor

python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

## Usage

**Run the app locally:**

```bash
cd app
streamlit run app.py
```

**Reproduce the full pipeline from scratch** (optional — trained model +
data are already included in the repo):

```bash
cd scripts
python scraper.py --pages 200 --output ../data/raw/cars_raw.csv
python clean.py
python features.py
python train.py
python tune_and_analyze.py   # optional: hyperparameter search + error analysis
```

## What I'd improve next

- Expand the make→model coverage for rare/luxury brands (more scraped
  listings, or a separate model for the luxury segment)
- Add trim/variant-level features for higher-end cars, where spec
  details matter more to price than for common commuter cars
- Track price history over time per listing to detect stale/relisted
  ads and improve the "fair price" comparison range

## Tech stack

- **Scraping:** `requests`, `BeautifulSoup4`
- **Data processing:** `pandas`, `numpy`
- **Modeling:** `scikit-learn`, `XGBoost`
- **App:** `Streamlit`
- **Deployment:** Streamlit Community Cloud

## License

MIT — see [LICENSE](LICENSE).

---

Built by [Muhammad Ali Ejaz](https://github.com/MuhammadAliEjaz1) — BS
Data Science student, Islamia University of Bahawalpur.
