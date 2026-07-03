# Predicting Train Delays on the Deutsche Bahn Network

Machine learning pipeline for predicting and explaining train delays on the Deutsche Bahn (DB) network, combining a longitudinal, multi-season operational dataset with a passenger survey to connect model-driven delay predictors to what travelers actually experience as disruptive.

<!-- TODO: add badges once repo is live, e.g. Python version, license, last commit -->
<!-- ![Python](https://img.shields.io/badge/python-3.x-blue) ![License](https://img.shields.io/badge/license-MIT-green) -->

## Overview

Train delay on the DB network is not random — it is a structured, autoregressive, temporally and spatially patterned process. This project builds and evaluates thirteen machine learning models across multiple paradigms to predict delays, identifies which operational features drive those predictions, and triangulates the results against a passenger survey (n = 80) to see whether what the models flag as important matches what passengers say frustrates them most.

**Key finding:** technical predictors and passenger-perceived pain points largely align — except for one gap: communication quality during a delay isn't captured by any operational variable in the current dataset, despite being the top passenger complaint.

## Data

- **Primary dataset:** [`piebro/deutsche-bahn-data`](https://huggingface.co/datasets/piebro/deutsche-bahn-data) — polled operational data from the DB API. Licensed CC BY 4.0, contains no personally identifiable information.
- **Secondary dataset:** an original passenger survey (n = 80), collected under GDPR-compliant, anonymized, purposive sampling. Raw responses are **not included in this repository** — only aggregated/anonymized outputs are shared, per the project's ethics approval.

<!-- TODO: note the date range / number of records once you confirm exact figures from Chapter 3 -->

## Methodology — Six-Phase Pipeline

The project is implemented as a sequence of self-contained Jupyter notebooks, coordinated by a shared `config.json` (feature definitions, file paths, hyperparameters, evaluation thresholds) written by Phase 1 and consumed by every phase after it.

| Phase | Notebook | Purpose |
|---|---|---|
| 1 | `<!-- TODO -->` | Data acquisition & cleaning (four-step pipeline; filters ~50% empty-string ride identifiers) |
| 2 | `<!-- TODO -->` | Feature engineering (lag features, cyclical time encoding, spatial features) |
| 3 | `<!-- TODO -->` | Train/test partitioning (ride-based `GroupShuffleSplit`, fixing an earlier data-leakage issue from naive time-based splitting) |
| 4 | `<!-- TODO -->` | Model training (13 models: tree ensembles, linear/regularized regression, clustering, deep learning) |
| 5 | `<!-- TODO -->` | Evaluation & explainability (SHAP-based feature importance) |
| 6 | `<!-- TODO -->` | Survey analysis & triangulation against model findings |

## Models

<!-- TODO: fill in the actual 13 models and headline metrics (RMSE/MAE/R² etc.) once results table is finalized -->

| Model | Type | Metric | Score |
|---|---|---|---|
| XGBoost | Gradient boosting | — | — |
| LightGBM | Gradient boosting | — | — |
| Ridge / Lasso | Regularized linear | — | — |
| ... | ... | — | — |

## Repository Structure

```
<!-- TODO: replace with your real structure, e.g. -->
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_train_test_split.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_evaluation_explainability.ipynb
│   └── 06_survey_analysis.ipynb
├── config.json.example
├── requirements.txt
├── outputs/
│   └── figures, metrics, SHAP plots
├── README.md
└── LICENSE
```

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

<!-- TODO: add any non-pip dependencies, Python version requirement -->

## Usage

```bash
<!-- TODO: e.g. -->
cp config.json.example config.json   # fill in your local paths
jupyter notebook notebooks/01_data_cleaning.ipynb
```

Run the notebooks in numerical order — each phase depends on outputs from the previous one via `config.json`.

## Limitations

- The survey sample (n = 80) is not representative of DB's full passenger base (~5 million daily journeys) — findings from it are indicative, not generalizable.
- Hyperparameters were selected via early stopping and cross-validated search rather than exhaustive grid or Bayesian optimization.
- Non-probability purposive sampling precludes formal statistical generalization of the survey's thematic and Likert-scale findings.

## Recommendations for Future Work

1. Capture communication quality as a modeled variable (e.g. via social media complaint data or CSAT scores).
2. Deploy and evaluate the trained classifiers in a live, real-time context against DB's API.
3. Extend the dataset to five or more years for robust seasonal modeling and full annual-cycle cross-validation.
4. Adapt the six-phase pipeline to other European rail networks (ProRail, Network Rail, SNCF) for international benchmarking.

## Ethics & Compliance

This research was conducted under GDPR (Regulation (EU) 2016/679) and the ethical guidelines of the Berlin School of Business and Innovation (BSBI) and the University for the Creative Arts (UCA). No personally identifiable survey data is stored or published in this repository.

## Citation

<!-- TODO: add your preferred citation format, e.g. -->
```
Tabikh, S. (2026). Predicting Train Delays on the Deutsche Bahn Network [Dissertation].
Berlin School of Business and Innovation / University for the Creative Arts.
```

## License

<!-- TODO: confirm your institution's IP policy before choosing a license (MIT / Apache-2.0 are common defaults) -->

## Author

**Sajed Tabikh**
<!-- TODO: add LinkedIn / contact / portfolio link -->
