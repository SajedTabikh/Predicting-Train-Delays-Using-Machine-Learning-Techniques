# Predicting Train Delays Using Machine Learning Techniques
### A Case Study on Deutsche Bahn Operational Data

MSc Data Analytics dissertation project — a six-phase machine learning pipeline that predicts train delays across the Deutsche Bahn (DB) network, explains what drives them, and cross-checks the findings against a passenger survey to see whether what the models flag as important matches what passengers actually experience as disruptive.

**Author:** Sajed Tabikh · MSc Data Analytics · 2026

---

## Overview

Train delay on the DB network is not random — it is a structured, autoregressive, temporally and spatially patterned process that machine learning can model with practically useful accuracy.

Using a longitudinal dataset of **893,542 stop-level observations spanning 21 months (July 2024 – March 2026)**, this project applies six analytical phases — regression, multiclass classification, LSTM/Transformer sequence modelling, unsupervised anomaly detection, K-Means geospatial clustering, and consolidated SHAP-based feature importance — to the same unified dataset under a standardized evaluation protocol. A primary passenger survey (n = 80) provides complementary evidence, tested with a Kruskal-Wallis H test and a Spearman rank correlation.

**Headline result:** the best regression model (MLP Neural Network) achieves **MAE = 0.913 minutes**, a **70.7% improvement** over a mean-baseline predictor, with 97.2% of predictions falling within 5 minutes of the true delay — accurate enough for operational, passenger-facing alerting.

**Key gap identified:** technical predictors and passenger-perceived pain points largely align — except for one. Poor communication during a delay is passengers' top-cited frustration (63.8% agreement), but no operational variable in the current dataset captures communication quality, meaning it's invisible to every model trained here.

---

## Data

| | |
|---|---|
| **Source** | [`piebro/deutsche-bahn-data`](https://huggingface.co/datasets/piebro/deutsche-bahn-data) (Hugging Face, CC BY 4.0) |
| **Date range** | July 2024 – March 2026 (21 monthly Parquet files) |
| **Raw records** | ~105 million API polling events |
| **Clean stop-level observations** | 893,542 (after a four-step cleaning pipeline — ~50% of raw polling records carried empty-string ride identifiers and were filtered) |
| **Train / test split** | Ride-based `GroupShuffleSplit` (scikit-learn, `random_state=42`), zero ride overlap, verified programmatically before every phase |
| **Train set** | 715,232 stops · 114,497 unique rides (80%) |
| **Test set** | 178,310 stops · 28,625 unique rides (20%) |
| **Features** | 60 engineered variables across 9 feature groups (lag, temporal, weather, congestion, station statistics, geographic, encoded IDs, route, other) |
| **Secondary data** | Original passenger survey, n = 80, collected at four DB terminal stations (Berlin, München, Hamburg, Frankfurt Hbf) — GDPR-compliant, anonymized, non-probability purposive sample. Raw responses are **not included** in this repository; only aggregated outputs are shared. |

> A naive time-based train/test split was found during development to leak near-perfect autocorrelated lag information into the test set. This was corrected by adopting ride-based `GroupShuffleSplit` partitioning — the split strategy used throughout the pipeline.

---

## Methodology — Six-Phase Pipeline

Implemented as a sequence of self-contained Jupyter notebooks, coordinated by a shared `config.json` (feature definitions, file paths, hyperparameters, evaluation thresholds) written by Phase 1 and consumed by every phase after it. A `LEAKAGE_COLS` guard is verified programmatically at the start of each phase to confirm no within-ride target leakage.

| Phase | Task | What it does |
|---|---|---|
| **1 — Data Engineering** | Ingestion & feature construction | Loads 21 monthly Parquet files → four-step cleaning → geocoding → weather feature engineering → ride-based `GroupShuffleSplit` (seed=42) → lag & congestion features → exports train/test Parquets + `config.json` |
| **2 — Regression** | Predict delay in minutes | 8 models: Linear, Ridge, Lasso, Random Forest, XGBoost, LightGBM, MLP |
| **3 — Classification** | Predict delay severity (4-class) | 5 models: Logistic Regression, Random Forest, XGBoost, MLP (Softmax), LightGBM — inverse-frequency class weighting |
| **4 — Deep Learning & Geospatial** | Sequence modelling, anomaly detection, clustering | LSTM & Transformer delay-propagation models, LSTM Autoencoder anomaly detection, K-Means geospatial clustering, station hotspot mapping |
| **5 — Interpretability** | Explain the models | SHAP values (global + local), consolidated feature importance across 3 models, partial dependence plots, delay heatmaps |
| **6 — Survey Integration & Synthesis** | Connect models to passenger experience | Master model comparison, Kruskal-Wallis & Spearman tests, methodological triangulation, thesis synthesis |


---

## Results

### Phase 2 — Regression (delay in minutes, n = 178,310 held-out)

| Model | MAE (min) | RMSE (min) | R² | Directional Acc. | Within 5 min |
|---|---|---|---|---|---|
| Mean Baseline | 3.114 | 5.637 | 0.000 | 89.7% | 91.2% |
| Linear Regression | 1.113 | 2.774 | 0.758 | 96.3% | 97.1% |
| Ridge Regression | 1.104 | 2.768 | 0.759 | 96.3% | 97.2% |
| Lasso Regression | 1.103 | 2.768 | 0.759 | 96.3% | 97.2% |
| Random Forest | 1.055 | 2.691 | 0.772 | 96.6% | 96.9% |
| XGBoost | 1.044 | 2.693 | 0.772 | 96.4% | 97.2% |
| LightGBM | 1.026 | 2.660 | **0.777** | 96.4% | 97.0% |
| **MLP Neural Network ★** | **0.913** | 2.792 | 0.755 | **96.5%** | 97.2% |

★ Best MAE. LightGBM achieves the best R² (0.777). Lasso eliminates 7 of 57 candidate features, retaining 13 of 15 weather features as informative.

### Phase 3 — Classification (4-class delay severity, n = 178,310 held-out)

Class distribution: on-time/early 44.3%, slight delay 43.4%, moderate delay 8.8%, severe delay 3.5%.

| Model | Accuracy | Macro F1 | ROC-AUC | F1 (Severe) |
|---|---|---|---|---|
| Majority Baseline | 44.29% | 0.154 | 0.500 | 0.000 |
| Logistic Regression (OvR) | 73.33% | 0.706 | 0.885 | 0.669 |
| XGBoost | 77.74% | 0.755 | 0.932 | 0.736 |
| MLP Neural Network | — | 0.725 | 0.921 | — |
| **LightGBM ★** | **79.20%** | **0.779** | **0.938** | **0.792** |

★ Best Macro F1 — a 407.7% improvement over the majority-class baseline. LightGBM correctly identifies 80.1% of all genuinely severe (>15 min) delays.

### Phase 4 — Deep Learning & Geospatial

- **LSTM Sequence Model** (5-stop look-back, 600,735 training sequences): MAE = 0.962 min, RMSE = 2.442 min, **R² = 0.826** — best R² of any model, though trailing the MLP on MAE.
- **Transformer Encoder**: near-identical performance (MAE = 0.957 min, R² = 0.826).
- **LSTM Autoencoder (anomaly detection):** flags 5.4% of test sequences as anomalous (reconstruction error > 95th percentile); anomalous stops average 6.10 min delay vs. 2.61 min for normal stops (2.34× elevation).
- **K-Means geospatial clustering** (5,272 stations, K=2, Silhouette = 0.425): identifies a central-eastern German corridor with the highest mean delays and severe-delay proportion. Top hotspots: Wolfsburg Hbf (7.64 min mean), Köln Hbf (5.60–6.50 min), Frankfurt (Main) Hbf (4.7–5.7 min).

### Phase 5 — Feature Importance

Consolidated (normalized, averaged) across XGBoost Regressor, XGBoost Classifier, and LightGBM Classifier:

| Rank | Feature | Group |
|---|---|---|
| 1 | `prev_stop_was_delayed` | Lag |
| 2 | `prev_stop_delay` | Lag |
| 3 | `final_destination_station_enc` | Encoded ID |
| 4 | `delay_rolling3` | Lag |
| 5 | `temp` | Weather |
| 6 | `congestion_ratio` | Congestion |
| 7 | `station_name_enc` | Station |
| 8 | `lon` / `lat` | Geo |
| 9 | `station_std` | Station |
| 10 | `stop_number` | Route |

Lag features carry the highest summed group importance (1.719), confirming strong serial dependence of delay within a ride, followed by temporal (1.452), station statistics (1.140), weather (0.999), and geography (0.654).

### Phase 6 — Survey Findings (n = 80)

- 88.8% agree punctuality is their top priority; only 16.3% are satisfied with current DB punctuality.
- 63.8% agree DB's communication during a delay is poor — the largest unaddressed gap identified in this dissertation.
- 71.3% would use an ML-based delay prediction app; 85.0% say real-time alerts would help them plan.
- **Kruskal-Wallis H test:** perceived severity of long delays differs significantly by travel purpose (H(2) = 9.32, p = 0.0094, η² = 0.118, medium effect) — commuters report the highest severity ratings (4.91/5), followed by business (4.61/5) and leisure travelers (4.23/5).
- **Spearman rank correlation:** travel frequency vs. satisfaction, ρ = −0.412, p = 0.0018.

---

## Repository Structure

```
├── notebooks/
│   ├── 01_phase1_data_engineering.ipynb
│   ├── 02_phase2_regression.ipynb
│   ├── 03_phase3_classification.ipynb
│   ├── 04_phase4_deep_learning_geospatial.ipynb
│   ├── 05_phase5_interpretability.ipynb
│   └── 06_phase6_survey_synthesis.ipynb
├── config.json.example      # copy to config.json and fill in local paths
├── data/                    # gitignored — see Data section for source
├── outputs/                 # figures, metrics, SHAP plots, model artifacts
├── requirements.txt
├── README.md
└── LICENSE
```


---

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

`requirements.txt` should pin at least:

```
pandas>=2.0
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
tensorflow>=2.12
shap>=0.42
```

## Usage

```bash
cp config.json.example config.json   # set your local data paths
jupyter notebook notebooks/01_phase1_data_engineering.ipynb
```

Run notebooks in numerical order — each phase reads the `config.json` and Parquet outputs written by Phase 1, and later phases depend on model artifacts saved by earlier ones.

---

## Limitations

- The passenger survey (n = 80) is not statistically representative of DB's full passenger population (~5 million daily journeys); findings from it should be treated as indicative rather than generalizable.
- Non-probability purposive sampling precludes formal generalization of the thematic and Likert-scale survey findings beyond the respondent group.
- Hyperparameters across all 13 trained models were selected via early stopping and cross-validated search rather than exhaustive grid or Bayesian optimization.
- Lag features are necessarily set to zero at each ride's first stop (no prior within-ride delay history), slightly understating predictions at journey origins.
- Station-level (not sensor-level) weather data introduces spatial imprecision in regions of complex terrain (e.g. Alpine foothills).

## Recommendations for Future Work

1. Capture delay-communication quality as a modeled variable (e.g. via social media complaint data or CSAT scores) — the single largest gap between what the models predict and what passengers say matters most.
2. Deploy the trained classifiers against DB's live API to evaluate real-world impact of severe-delay alerts.
3. Extend the dataset beyond 21 months to enable formal seasonal cross-validation across complete annual cycles.
4. Adapt the six-phase pipeline to other European rail networks (ProRail, Network Rail, SNCF) for the first rigorous international benchmark of ML delay-prediction performance.

---

## Ethics & Compliance

This research was conducted in full compliance with the General Data Protection Regulation (GDPR, Regulation (EU) 2016/679) and the ethical guidelines of the Berlin School of Business and Innovation (BSBI) and the University for the Creative Arts (UCA).

- The operational dataset (`piebro/deutsche-bahn-data`) is public, CC BY 4.0 licensed, and contains no personally identifiable information.
- Survey participation was voluntary; no personally identifiable data was collected; responses are retained only in anonymized, aggregated form and will be deleted within 12 months of dissertation submission per BSBI data-retention guidelines.

## Citation

```
Tabikh, S. (2026). Predicting Train Delays Using Machine Learning Techniques:
A Case Study on Deutsche Bahn Operational Data. MSc Data Analytics dissertation,
Berlin School of Business and Innovation / University for the Creative Arts.
```

## License

This repository is shared for academic and portfolio purposes. Before opening it under a permissive license (e.g. MIT), check BSBI/UCA's IP policy on dissertation code — some institutions require an embargo period before public release.

## Author

**Sajed Tabikh** — MSc Data Analytics, 2026
