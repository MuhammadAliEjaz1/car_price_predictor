# 🚗 PakWheels Used Car Price Predictor

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-in%20progress-yellow.svg)

Predicts fair market price for used cars in Pakistan, using data scraped
directly from [PakWheels.com](https://www.pakwheels.com) — Pakistan's
largest used car marketplace.

Unlike generic car price predictors (almost all trained on US/European
markets), this project models **Pakistan-specific pricing dynamics** that
standard datasets don't capture:

- **Import/assembly status** — locally assembled vs. imported (JDM/duty
  status) cars follow very different price curves in Pakistan
- **Registered-in city premium** — cars registered in major cities
  (Karachi, Lahore, Islamabad) command different resale prices than the
  same car registered elsewhere
- **Listing city vs. registration city gap** — the mismatch between where
  a car is listed and where it's registered is itself a useful signal

## Table of Contents

- [Why this project](#why-this-project)
- [Project layout](#project-layout)
- [Setup](#setup)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Tech stack](#tech-stack)
- [License](#license)

## Why this project

Most "car price prediction" portfolio projects reuse the same handful of
public Kaggle datasets (largely US/German market data) and don't reflect
how used cars are actually priced in Pakistan. This project scrapes fresh,
current listings and engineers features specific to the local market, so
the model — and the "is this a fair price?" tool built on top of it —
actually reflects how Pakistani buyers and sellers price cars.

## Project layout

```
pakwheels-price-predictor/
├── data/
│   ├── raw/              # scraper output (cars_raw.csv, etc.) — gitignored
│   └── processed/        # cleaned/feature-engineered datasets — gitignored
├── scripts/
│   └── scraper.py        # two-stage scraper (search pages + detail pages)
├── notebooks/            # EDA, cleaning, and modeling notebooks
├── models/                # trained model artifacts — gitignored
├── .github/workflows/     # CI (lint)
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/pakwheels-price-predictor.git
cd pakwheels-price-predictor

python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

## Usage

**Scrape listings:**

```bash
cd scripts
python scraper.py --pages 20 --output ../data/raw/cars_raw.csv
```

**Resume a scrape** (skips listing IDs already in the file):

```bash
python scraper.py --pages 20 --output ../data/raw/cars_raw.csv --resume
```

**Quick test without the slower detail-page pass** (no `assembly` /
`registered_in` columns, but much faster):

```bash
python scraper.py --pages 5 --output ../data/raw/test.csv --no-details
```

Run `python scraper.py --help` for all options.

> The scraper rate-limits itself (default 1.5s + jitter between requests).
> Please don't lower this aggressively — be a good citizen when scraping
> PakWheels.

## Roadmap

- [x] Two-stage scraper (search pages + detail pages)
- [ ] Full data collection (target: 5,000–8,000 listings)
- [ ] Data cleaning + make/model parsing from listing titles
- [ ] Feature engineering (price/cc, city premium, assembly effect)
- [ ] Model training and evaluation (XGBoost/LightGBM, MAE in PKR)
- [ ] FastAPI backend + React frontend demo ("is this a fair price?" tool)
- [ ] Deployment (Hugging Face Spaces / Vercel)

## Tech stack

- **Scraping:** `requests`, `BeautifulSoup4`, `lxml`
- **Data processing:** `pandas`, `numpy`
- **Modeling:** `scikit-learn`, `XGBoost`
- **API:** `FastAPI`
- **Frontend:** React *(planned)*

## License

MIT — see [LICENSE](LICENSE).

---

Built by [Muhammad Ali Ejaz](https://github.com/<your-username>) — BS Data
Science student, Islamia University of Bahawalpur.
