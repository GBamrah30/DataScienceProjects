# NYC Motor Vehicle Collisions — End-to-End Data Science Project

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red) ![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

## Overview

An end-to-end data science project analyzing NYC motor vehicle collision data to surface actionable safety insights for city planners and DOT analysts. The project covers the full data science workflow: ETL pipeline, SQL database design, exploratory data analysis, geospatial visualization, an interactive Streamlit dashboard, and machine learning models to predict collision severity.

**Built for:** NYC DOT safety analysts and city planners who need to identify which intersections to prioritize for safety interventions.

**Live Dashboard:** *(coming soon)*

**Dataset:** [NYC Motor Vehicle Collisions — NYC Open Data](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95)

---

## Key Findings

*(To be updated as analysis progresses)*

- Finding 1
- Finding 2
- Finding 3

---

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3.11+ |
| Data Processing | pandas, NumPy |
| Database | PostgreSQL |
| Visualization | Plotly, Streamlit |
| Machine Learning | scikit-learn, XGBoost |
| External Data | NOAA Weather API |
| Version Control | Git, GitHub |

---

## Project Structure

```
urban_infrastructure_project/
│
├── data/
│   ├── raw/                    # Original unmodified source data
│   ├── cleaned/                # Transformed, analysis-ready data
│
├── etl/
│   ├── extract.py              # Data extraction from NYC Open Data
│   ├── transform.py            # Cleaning and transformation logic
│   ├── load.py                 # Load to PostgreSQL
│   ├── weather_extract.py      # NOAA weather API pull and join
│
├── database/
│   ├── schema.sql              # Table definitions and relationships
│   ├── connection.py           # DB connection config
│
├── analysis/
│   ├── eda.py                  # Exploratory data analysis
│   ├── feature_engineering.py  # Feature creation for ML
│   ├── ml_models.py            # Model training and evaluation
│   ├── model_card.md           # Model documentation and limitations
│
├── app/
│   ├── streamlit_app.py        # Main dashboard entry point
│   ├── pages/                  # Individual dashboard pages
│
├── notebooks/                  # Exploratory Jupyter notebooks
│
├── requirements.txt
└── README.md
```

---

## Phase Roadmap

- [x] Project setup and environment configuration
- [ ] **Phase 1** — Data collection and ETL pipeline
- [ ] **Phase 2** — EDA, visualization, and baseline ML
- [ ] **Phase 3** — Streamlit dashboard
- [ ] **Phase 4** — Advanced machine learning
- [ ] **Phase 5** — Deployment and portfolio polish

---

## Dashboard Features *(Phase 3)*

- KPI overview — total collisions, injuries, fatalities, year-over-year change
- Filters by borough, date range, contributing factor, and weather condition
- Interactive collision heatmaps and borough comparison maps
- Top 20 high-risk intersection prioritization view
- Severity analysis by road user type (pedestrian, cyclist, motorist)
- Weather condition correlation with collision rates

---

## Machine Learning *(Phase 4)*

**Goal:** Predict collision severity and surface prioritized intersection recommendations for safety planners.

Models evaluated:
- Logistic Regression (baseline)
- Random Forest
- XGBoost

Key features include time of day, weather conditions, borough, contributing factor, vehicle type, and engineered risk scores.

See [`analysis/model_card.md`](analysis/model_card.md) for full model documentation, performance metrics, known limitations, and bias analysis.

---

## What I'd Build With More Time

- Real-time collision feed integration via live NYC Open Data API
- MTA delay data overlay to correlate transit disruptions with collision spikes
- Pedestrian foot traffic data to normalize collision rates by exposure
- 311 complaint data join to identify areas with known infrastructure issues
- Ensemble model improvements and hyperparameter tuning
- Docker containerization for reproducible deployment
- FastAPI backend for serving model predictions

---

## Setup & Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/urban-infrastructure-project.git
cd urban-infrastructure-project

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Set up PostgreSQL connection
# Update database/connection.py with your credentials

# Run ETL pipeline
python etl/extract.py
python etl/transform.py
python etl/load.py

# Launch dashboard
streamlit run app/streamlit_app.py
```

---

## Author

**Your Name**
[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)
