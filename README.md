# GROUP02_CENG313
Jane Street Real-Time Market Data Forecasting 

## Project Overview

- **Course:** CENG313 - Data Science
- **Kaggle Competition:** [Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting)
- **GitHub Repository:** `GROUP02_CENG313`
- **GitHub Organization:** [ceng313-2026](https://github.com/ceng313-2026)

This project focuses on predicting financial market responses (`responder_6`) using historical market data provided by Jane Street. Due to the dataset's scale, high noise-to-signal ratio and non-stationary nature of financial time-series data, the project uses memory-conscious data inspection and preprocessing, chronological validation, and machine-learning regression models.

## Team Members

| Name Surname | Student ID | GitHub Username |
|---|---:|---|
| Rümeysa Demirelli | 24118080023 | [@godsuyasan](https://github.com/godsuyasan) |
| Emine Yılmaz | 23118080050 | [@emiyilm](https://github.com/emiyilm) |
| Hatice Kübra Küskü | 24118080403 | [@hkubra00](https://github.com/hkubra00) |

## Kaggle Competition and Dataset

The selected competition is [Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting). The dataset contains multi-partitioned Apache Parquet files (`train.parquet`, `test.parquet`, and `lags.parquet`) and metadata CSV files (`features.csv` and `responders.csv`).

### Dataset Summary

- **Total rows audited:** 47,127,338 observations across 10 Hive partitions (`partition_id=0..9`).
- **Temporal span:** 1,699 unique trading days (`date_id` from `0` to `1698`) and 968 intraday time steps (`time_id` from `0` to `967`).
- **Assets:** 39 unique financial instruments (`symbol_id`).
- **Features:** 79 anonymized feature columns:
  - 76 continuous features (`Float32`).
  - 3 discrete/categorical features (`feature_09`, `feature_10`, and `feature_11`).
- **Targets and weights:** 9 responder columns (`responder_0` to `responder_8`). The primary competition target is `responder_6`; observations are evaluated using the provided `weight` column.

> **Data policy:** Dataset files are not included in this repository. Competition data should be accessed through Kaggle and credentials should be configured when running the notebook.

## Assignment 1: EDA and Baseline Implementations

### Exploratory Data Analysis

- **Missing-value audit:** Exact null counts were extracted from Parquet row-group metadata using `pyarrow.parquet`, avoiding the need to load the complete dataset into memory.
  - 32 features had no missing values.
  - 34 features had a missing-value ratio between 0% and 5%.
  - 13 features had a missing-value ratio between 5% and 20%.
  - No features had more than 20% missing values. The highest reported missingness was 17.90%.
- **Target and weight analysis:** The distribution, empirical quantiles, skewness, kurtosis, and boundary values of `responder_6` were examined.

### Validation Strategy for the Closed Competition

The Jane Street competition is treated as a closed competition for this course project. Since a standard Kaggle late submission is not available to us, we evaluate the baseline models locally using a chronological **purged held-out split**, in line with the course instructions.

The dataset contains 1,699 trading days (`date_id` 0–1698). The final chronological block is reserved for evaluation:

| Partition | Trading days (`date_id`) | Purpose |
|---|---:|---|
| Training period | 851–1350 (500 days) | Data available for fitting the baseline models |
| Purge gap | 1351–1355 (5 days) | Excluded to reduce temporal leakage between training and evaluation |
| Held-out evaluation period | 1356–1698 (343 days) | Reserved for local validation/evaluation; not used to fit the models |

The held-out period is kept separate from model fitting. For the baseline experiments, due to Google Colab memory constraints, the notebook uses a sample of 500,000 training observations and 100,000 validation observations from the corresponding training and held-out periods.

In addition to the final held-out evaluation period, a four-fold chronological expanding/sliding-window scheme was designed with a maximum training window of 500 days and a 5-day purge gap:

- **Fold 1:** Train `0–333`; purge gap; validation `339–677`.
- **Fold 2:** Train `173–672`; purge gap; validation `678–1016`.
- **Fold 3:** Train `512–1011`; purge gap; validation `1017–1355`.
- **Fold 4:** Train `851–1350`; purge gap; held-out evaluation `1356–1698`.

### Preprocessing

- Training means and standard deviations are calculated in batches.
- Extreme standardized values are clipped to the range `[-10.0, +10.0]`.
- Missing values are imputed with `0.0` in standardized (Z-score) space.
- Preprocessed data is streamed to Parquet using Polars `sink_parquet` to reduce memory usage.

### Reusable Code

Reusable classes and helper functions are maintained in the `src/` directory and imported by the notebooks:

- `src/data.py`: chronological data splitting and preprocessing utilities.
- `src/models.py`: baseline model classes, including Zero, Ridge, and LightGBM wrappers (and additional model wrappers where implemented).
- `src/utils.py`: the competition's weighted R² metric and memory-conscious Parquet loading utilities.

### Baseline Models and Results

The baseline models are evaluated using the competition's weighted zero-mean R² metric:

\[
R^2_{\mathrm{weighted}} =
1 - \frac{\sum_i w_i (y_i-\hat{y}_i)^2}
{\sum_i w_i y_i^2}
\]

Due to Colab RAM limitations, the last 500,000 rows of the processed training data and the last 100,000 rows of the processed validation data from Fold 4 were loaded for baseline evaluation.
The reported experiments use 500,000 training observations and 100,000 held-out validation observations.

| Model | Train Weighted R² | Validation Weighted R² | Training and inference time | |
|---|---:|---:|---:|---|
| Zero Baseline | 0.000000 | 0.000000 | 0.01 s | Predicts zero for every observation. |
| Ridge Regression (`alpha=100`) | 0.009488 | -0.005350 | 0.80 s | Training score is positive, but held-out performance is negative. |
| LightGBM Regressor | 0.033057 | **0.002207** | 15.37 s | Achieves a positive held-out score among the tested baselines. |

**Main observations:**

1. Ridge Regression has a positive training score but a negative validation score, indicating limited generalization in this temporal setting.
2. LightGBM achieves a positive held-out Weighted R² of `0.002207` and performs best among the reported baselines.
3. The LightGBM split-importance analysis identifies `feature_06`, `feature_07`, `feature_04`, `feature_60`, `feature_05`, and `feature_36` as the most important features in the reported experiment.

## Assignment 2: Proposed Solution

This section will be updated when the proposed solution has been implemented and evaluated. It will describe the proposed method, its motivation, implementation details, validation setup, and results.

## Repository Structure

```text
GROUP02_CENG313/
├── README.md
├── assignment1_eda_baseline.ipynb
├── assignment2_solution.ipynb
└── src/
    ├── data.py
    ├── models.py
    └── utils.py
```

The repository must not contain downloaded competition data or other large data files. Reusable code belongs in `src/`, rather than being duplicated only inside notebook cells.

## How to Run the Notebooks

### Assignment 1

1. Open `assignment1_eda_baseline.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Configure Kaggle credentials in Colab Secrets using the names expected by the notebook (for example, `KAGGLE_USERNAME` and `KAGGLE_KEY`).
3. Ensure the required Python packages are installed. The notebook includes the relevant setup steps.
4. Run the notebook cells in order. The notebook audits the data, performs preprocessing, imports reusable components from `src/`, and runs the baseline experiments.

A T4 GPU runtime may be selected in Colab if required by the notebook. The baseline implementations should be run using the runtime configuration documented in the notebook.

### Assignment 2

Instructions for running the proposed solution will be added here when Assignment 2 is implemented.

## Progress Logs

- **October 2026:** Formed Group 02, selected the Jane Street Real-Time Market Data Forecasting competition and initialized the project repository.
- **November 10, 2026:** Implemented Kaggle data access, inspected the dataset schema with Polars and performed a memory-conscious missing-value audit using PyArrow metadata.
- **November 11, 2026:** Analyzed the `responder_6` and `weight` distributions, designed chronological purged splits and prepared the Fold 4 data.
- **November 12–13, 2026:** Implemented reusable utilities and baseline model components; evaluated Zero, Ridge, and LightGBM baselines; documented the held-out validation strategy for the closed competition.

## Kaggle Submission History

No Kaggle submission is reported for Assignment 1. The competition is handled as a closed competition for this course project, so the baseline results are reported using the local purged held-out evaluation strategy described above in "**Validation Strategy for the Closed Competition**".
